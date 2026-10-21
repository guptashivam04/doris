// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "storage/compaction/collection_statistics.h"

#include <array>
#include <set>
#include <sstream>
#include <stack>

#include "common/exception.h"
#include "core/column/column_vector.h"
#include "core/data_type/data_type_factory.hpp"
#include "exprs/vexpr.h"
#include "exprs/vexpr_context.h"
#include "exprs/vliteral.h"
#include "exprs/vslot_ref.h"
#include "storage/index/index_file_reader.h"
#include "storage/index/index_reader_helper.h"
#include "storage/index/inverted/analyzer/analyzer.h"
#include "storage/index/inverted/similarity/bm25_similarity.h"
#include "storage/index/inverted/util/string_helper.h"
#include "storage/index/inverted/util/term_iterator.h"
#include "storage/index/zone_map/zone_map_index.h"
#include "storage/iterators.h"
#include "storage/rowset/beta_rowset.h"
#include "storage/rowset/rowset.h"
#include "storage/rowset/rowset_reader.h"
#include "storage/segment/column_reader.h"
#include "storage/segment/segment.h"
#include "storage/segment/segment_loader.h"
#include "storage/tablet/tablet_meta.h"
#include "util/uid_util.h"

namespace doris {
#include "common/compile_check_begin.h"

namespace collection_statistics_detail {

// How many postings scanning a term decodes in the time leaping costs per hidden row.
constexpr uint64_t kSkipsPerScannedPosting = 32;

uint64_t count_listed_docs(const segment_v2::TermIterator& postings,
                           std::span<const uint32_t> docs) {
    uint64_t listed = 0;
    int64_t doc = -1;
    size_t next = 0;
    while (next < docs.size()) {
        if (doc < docs[next]) {
            // skipTo always moves past the current document, so only leap forward.
            doc = postings.advance(cast_set<int32_t>(docs[next]));
            if (doc == INT_MAX) {
                break;
            }
        }
        if (doc == docs[next]) {
            ++listed;
            ++next;
        } else {
            next = static_cast<size_t>(
                    std::lower_bound(docs.begin() + next, docs.end(), static_cast<uint32_t>(doc)) -
                    docs.begin());
        }
    }
    return listed;
}

uint64_t count_listed_docs(const segment_v2::TermIterator& postings,
                           const std::vector<uint64_t>& doc_bits) {
    uint64_t listed = 0;
    const auto is_set = [&doc_bits](uint32_t doc) {
        const size_t word = doc / 64;
        return word < doc_bits.size() && ((doc_bits[word] >> (doc % 64)) & 1) != 0;
    };
    DocRange range;
    // A block without documents only comes from a corrupt image and ends the term.
    while (postings.read_range(&range) && range.doc_many_size_ != 0) {
        if (range.type_ == DocRangeType::kRange) {
            for (uint32_t doc = range.doc_range.first; doc < range.doc_range.second; ++doc) {
                listed += is_set(doc) ? 1 : 0;
            }
        } else {
            for (uint32_t i = 0; i < range.doc_many_size_; ++i) {
                listed += is_set((*range.doc_many)[i]) ? 1 : 0;
            }
        }
    }
    return listed;
}

std::vector<uint64_t> doc_bitset(std::span<const uint32_t> docs, uint32_t doc_count) {
    std::vector<uint64_t> bits((static_cast<size_t>(doc_count) + 63) / 64, 0);
    for (const uint32_t doc : docs) {
        bits[doc / 64] |= uint64_t {1} << (doc % 64);
    }
    return bits;
}

namespace {

// The length each encoded norm stands for.
const std::array<uint32_t, 256>& norm_length_table() {
    static const std::array<uint32_t, 256> table = [] {
        std::array<uint32_t, 256> lengths {};
        for (size_t norm = 0; norm < lengths.size(); ++norm) {
            lengths[norm] = static_cast<uint32_t>(
                    segment_v2::BM25Similarity::byte4_to_int(static_cast<uint8_t>(norm)));
        }
        return lengths;
    }();
    return table;
}

} // namespace

uint64_t sum_norm_lengths(const uint8_t* norms, std::span<const uint32_t> docs) {
    const auto& table = norm_length_table();
    uint64_t lengths = 0;
    for (const uint32_t doc : docs) {
        lengths += table[norms[doc]];
    }
    return lengths;
}

uint64_t sum_norm_lengths(const uint8_t* norms, uint32_t doc_count) {
    const auto& table = norm_length_table();
    uint64_t lengths = 0;
    for (uint32_t doc = 0; doc < doc_count; ++doc) {
        lengths += table[norms[doc]];
    }
    return lengths;
}

} // namespace collection_statistics_detail

namespace {

// The rows of a segment that a scan with `hidden_rows` does not return, ascending.
Status collect_hidden_docs(const RowsetSharedPtr& rowset, int32_t seg_id,
                           const TabletSchema& tablet_schema, const MowHiddenRows& hidden_rows,
                           const io::IOContext* io_ctx, std::vector<uint32_t>* hidden_docs) {
    roaring::Roaring hidden;
    if (hidden_rows.delete_bitmap != nullptr) {
        hidden = *hidden_rows.delete_bitmap->get_agg(
                {rowset->rowset_id(), cast_set<uint32_t>(seg_id), hidden_rows.version});
    }

    // A row written with the delete sign stays readable until base compaction removes it, while
    // queries filter it out by predicate.
    if (tablet_schema.delete_sign_idx() >= 0) {
        OlapReaderStatistics stats;
        SegmentCacheHandle cache_handle;
        RETURN_IF_ERROR(SegmentLoader::instance()->load_segment(
                std::static_pointer_cast<BetaRowset>(rowset), seg_id, &cache_handle, true, false,
                &stats, io_ctx));
        DORIS_CHECK(cache_handle.get_segments().size() == 1);
        const auto& segment = cache_handle.get_segments()[0];
        const TabletColumn& delete_sign = tablet_schema.column(tablet_schema.delete_sign_idx());

        bool may_hold_deletes = true;
        std::shared_ptr<segment_v2::ColumnReader> column_reader;
        auto status = segment->get_column_reader(delete_sign, &column_reader, &stats, io_ctx);
        if (status.is<ErrorCode::NOT_FOUND>()) {
            // A segment written before the column existed reads its default, 0.
            may_hold_deletes = false;
        } else {
            RETURN_IF_ERROR(status);
            if (column_reader != nullptr && column_reader->has_zone_map()) {
                segment_v2::ZoneMap zone_map;
                RETURN_IF_ERROR(column_reader->get_segment_zone_map(&zone_map));
                may_hold_deletes = zone_map.has_not_null && !zone_map.pass_all &&
                                   zone_map.max_value.get<TYPE_TINYINT>() != 0;
            }
        }
        if (may_hold_deletes) {
            StorageReadOptions read_options;
            read_options.stats = &stats;
            if (io_ctx != nullptr) {
                read_options.io_ctx = *io_ctx;
            }
            std::unique_ptr<segment_v2::ColumnIterator> iterator;
            RETURN_IF_ERROR(segment->new_column_iterator(delete_sign, &iterator, &read_options));
            segment_v2::ColumnIteratorOptions iterator_options {
                    .use_page_cache = !config::disable_storage_page_cache,
                    .file_reader = segment->file_reader().get(),
                    .stats = &stats,
                    .io_ctx = read_options.io_ctx,
            };
            RETURN_IF_ERROR(iterator->init(iterator_options));
            RETURN_IF_ERROR(iterator->seek_to_ordinal(0));
            auto data_type = DataTypeFactory::instance().create_data_type(delete_sign);
            constexpr size_t kBatch = 4096;
            for (uint32_t begin = 0; begin < segment->num_rows();) {
                size_t rows = std::min<size_t>(kBatch, segment->num_rows() - begin);
                MutableColumnPtr column = data_type->create_column();
                RETURN_IF_ERROR(iterator->next_batch(&rows, column));
                if (rows == 0) {
                    return Status::Corruption("segment {} ended after {} of its {} rows",
                                              segment->id(), begin, segment->num_rows());
                }
                const auto& signs = assert_cast<const ColumnInt8&>(*column).get_data();
                for (size_t i = 0; i < rows; ++i) {
                    if (signs[i] != 0) {
                        hidden.add(begin + cast_set<uint32_t>(i));
                    }
                }
                begin += cast_set<uint32_t>(rows);
            }
        }
    }

    hidden_docs->resize(hidden.cardinality());
    hidden.toUint32Array(hidden_docs->data());
    return Status::OK();
}

} // namespace

Status CollectionStatistics::collect(RuntimeState* state,
                                     const std::vector<RowSetSplits>& rs_splits,
                                     const TabletSchemaSPtr& tablet_schema,
                                     const VExprContextSPtrs& common_expr_ctxs_push_down,
                                     io::IOContext* io_ctx, const MowHiddenRows* hidden_rows) {
    std::unordered_map<std::wstring, CollectInfo> collect_infos;
    RETURN_IF_ERROR(
            extract_collect_info(state, common_expr_ctxs_push_down, tablet_schema, &collect_infos));
    if (collect_infos.empty()) {
        LOG(WARNING) << "Index statistics collection: no collect info extracted.";
        return Status::OK();
    }

    for (const auto& rs_split : rs_splits) {
        const auto& rs_reader = rs_split.rs_reader;
        auto rowset = rs_reader->rowset();
        auto num_segments = rowset->num_segments();
        for (int32_t seg_id = 0; seg_id < num_segments; ++seg_id) {
            auto status = process_segment(rowset, seg_id, tablet_schema.get(), collect_infos,
                                          hidden_rows, io_ctx);
            if (!status.ok()) {
                if (status.code() == ErrorCode::INVERTED_INDEX_FILE_NOT_FOUND ||
                    status.code() == ErrorCode::INVERTED_INDEX_BYPASS) {
                    LOG(ERROR) << "Index statistics collection failed: " << status.to_string();
                } else {
                    return status;
                }
            }
        }
    }

    // Build a single-line log with query_id, tablet_ids, and per-field term statistics
    if (VLOG_IS_ON(1)) {
        std::set<int64_t> tablet_ids;
        for (const auto& rs_split : rs_splits) {
            if (rs_split.rs_reader && rs_split.rs_reader->rowset()) {
                tablet_ids.insert(rs_split.rs_reader->rowset()->rowset_meta()->tablet_id());
            }
        }

        std::ostringstream oss;
        oss << "CollectionStatistics: query_id=" << print_id(state->query_id());

        oss << ", tablet_ids=[";
        bool first_tablet = true;
        for (int64_t tid : tablet_ids) {
            if (!first_tablet) oss << ",";
            oss << tid;
            first_tablet = false;
        }
        oss << "]";

        oss << ", total_num_docs=" << _total_num_docs;

        for (const auto& [ws_field_name, num_tokens] : _total_num_tokens) {
            oss << ", {field=" << StringHelper::to_string(ws_field_name)
                << ", num_tokens=" << num_tokens << ", terms=[";

            bool first_term = true;
            for (const auto& [term, doc_freq] : _term_doc_freqs.at(ws_field_name)) {
                if (!first_term) oss << ", ";
                oss << "(" << StringHelper::to_string(term) << ":" << doc_freq << ")";
                first_term = false;
            }
            oss << "]}";
        }

        VLOG(1) << oss.str();
    }

    return Status::OK();
}

Status CollectionStatistics::extract_collect_info(
        RuntimeState* state, const VExprContextSPtrs& common_expr_ctxs_push_down,
        const TabletSchemaSPtr& tablet_schema, CollectInfoMap* collect_infos) {
    DCHECK(collect_infos != nullptr);

    std::unordered_map<TExprNodeType::type, PredicateCollectorPtr> collectors;
    collectors[TExprNodeType::MATCH_PRED] = std::make_unique<MatchPredicateCollector>();
    collectors[TExprNodeType::SEARCH_EXPR] = std::make_unique<SearchPredicateCollector>();

    for (const auto& root_expr_ctx : common_expr_ctxs_push_down) {
        const auto& root_expr = root_expr_ctx->root();
        if (root_expr == nullptr) {
            continue;
        }

        std::stack<VExprSPtr> stack;
        stack.emplace(root_expr);

        while (!stack.empty()) {
            auto expr = stack.top();
            stack.pop();

            if (!expr) {
                continue;
            }

            auto collector_it = collectors.find(expr->node_type());
            if (collector_it != collectors.end()) {
                RETURN_IF_ERROR(
                        collector_it->second->collect(state, tablet_schema, expr, collect_infos));
            }

            const auto& children = expr->children();
            for (const auto& child : children) {
                stack.push(child);
            }
        }
    }

    LOG(INFO) << "Extracted collect info for " << collect_infos->size() << " fields";

    return Status::OK();
}

Status CollectionStatistics::process_segment(const RowsetSharedPtr& rowset, int32_t seg_id,
                                             const TabletSchema* tablet_schema,
                                             const CollectInfoMap& collect_infos,
                                             const MowHiddenRows* hidden_rows,
                                             io::IOContext* io_ctx) {
    auto seg_path = DORIS_TRY(rowset->segment_path(seg_id));
    auto rowset_meta = rowset->rowset_meta();

    auto idx_file_reader = std::make_unique<IndexFileReader>(
            rowset_meta->fs(),
            std::string {InvertedIndexDescriptor::get_index_file_path_prefix(seg_path)},
            tablet_schema->get_inverted_index_storage_format(),
            rowset_meta->inverted_index_file_info(seg_id), rowset_meta->tablet_id());
    RETURN_IF_ERROR(idx_file_reader->init(config::inverted_index_read_buffer_size, io_ctx));

    std::vector<uint32_t> hidden_docs;
    if (hidden_rows != nullptr) {
        RETURN_IF_ERROR(collect_hidden_docs(rowset, seg_id, *tablet_schema, *hidden_rows, io_ctx,
                                            &hidden_docs));
    }

    int32_t total_seg_num_docs = 0;
    std::vector<uint64_t> hidden_doc_bits;

    for (const auto& [ws_field_name, collect_info] : collect_infos) {
        lucene::search::IndexSearcher* index_searcher = nullptr;
        lucene::index::IndexReader* index_reader = nullptr;

#ifdef BE_TEST
        auto compound_reader = DORIS_TRY(idx_file_reader->open(collect_info.index_meta, io_ctx));
        auto* reader = lucene::index::IndexReader::open(compound_reader.get());
        auto searcher_ptr = std::make_shared<lucene::search::IndexSearcher>(reader, true);
        index_searcher = searcher_ptr.get();
        index_reader = index_searcher->getReader();
#else
        InvertedIndexCacheHandle inverted_index_cache_handle;
        auto index_file_key = idx_file_reader->get_index_file_cache_key(collect_info.index_meta);
        InvertedIndexSearcherCache::CacheKey searcher_cache_key(index_file_key);

        if (!InvertedIndexSearcherCache::instance()->lookup(searcher_cache_key,
                                                            &inverted_index_cache_handle)) {
            auto compound_reader =
                    DORIS_TRY(idx_file_reader->open(collect_info.index_meta, io_ctx));
            auto* reader = lucene::index::IndexReader::open(compound_reader.get());
            size_t reader_size = reader->getTermInfosRAMUsed();
            auto searcher_ptr = std::make_shared<lucene::search::IndexSearcher>(reader, true);
            auto* cache_value = new InvertedIndexSearcherCache::CacheValue(
                    std::move(searcher_ptr), reader_size, UnixMillis());
            InvertedIndexSearcherCache::instance()->insert(searcher_cache_key, cache_value,
                                                           &inverted_index_cache_handle);
        }

        auto searcher_variant = inverted_index_cache_handle.get_index_searcher();
        auto index_searcher_ptr = std::get<FulltextIndexSearcherPtr>(searcher_variant);
        index_searcher = index_searcher_ptr.get();
        index_reader = index_searcher->getReader();
#endif
        const int32_t index_doc_count = index_reader->maxDoc();
        if (!hidden_docs.empty() && hidden_docs.back() >= static_cast<uint32_t>(index_doc_count)) {
            return Status::Error<ErrorCode::INVERTED_INDEX_FILE_CORRUPTED>(
                    "hidden row {} of segment {} is beyond the {} documents of its index",
                    hidden_docs.back(), seg_path, index_doc_count);
        }
        const int32_t visible_doc_count = index_doc_count - cast_set<int32_t>(hidden_docs.size());
        total_seg_num_docs = std::max(total_seg_num_docs, visible_doc_count);

        // BM25 on an analyzed index needs the record length of every row, and CLucene keeps
        // them, together with the field's token count, in the norms. A segment written without
        // norms would feed a zero avgdl, or rank its rows as zero-length documents next to the
        // segments that have norms, so refuse to score the collection. Keyword indexes and
        // built-in lowercase normalizers can use unit frequency and length instead.
        const auto token_count = index_reader->sumTotalTermFreq(ws_field_name.c_str());
        const bool single_term_normalizer = segment_v2::IndexReaderHelper::is_normalizer_only(
                collect_info.index_meta->properties());
        if (!token_count.has_value() && !single_term_normalizer &&
            segment_v2::inverted_index::InvertedIndexAnalyzer::should_analyzer(
                    collect_info.index_meta->properties())) {
            return Status::Error<ErrorCode::INVERTED_INDEX_NOT_SUPPORTED>(
                    "BM25 scoring requires norms, but segment {} was written without norms for "
                    "field {}. Norms are left out when the index sets \"norms\" = \"false\" or, "
                    "for a variant path, when inverted_index_skip_norms_for_variant is on",
                    seg_path, StringHelper::to_string(ws_field_name));
        }
        if (hidden_rows == nullptr) {
            // Missing term totals imply one token per row for keyword and lowercase indexes.
            _total_num_tokens[ws_field_name] += token_count.value_or(index_reader->maxDoc());
        } else if (!token_count.has_value()) {
            _total_num_tokens[ws_field_name] += static_cast<uint64_t>(visible_doc_count);
        } else {
            // The exact token count of the hidden rows is not stored anywhere, and norms hold
            // lengths only approximately, so count the lengths the norms of the returned rows
            // stand for. A replica that compacted the hidden rows away counts the same rows
            // with the same norms.
            const uint8_t* norms = index_reader->norms(ws_field_name.c_str());
            _total_num_tokens[ws_field_name] +=
                    collection_statistics_detail::sum_norm_lengths(
                            norms, static_cast<uint32_t>(index_doc_count)) -
                    collection_statistics_detail::sum_norm_lengths(norms, hidden_docs);
        }

        for (const auto& term_info : collect_info.term_infos) {
            auto iter = TermIterator::create(io_ctx, false, index_reader, ws_field_name,
                                             term_info.get_single_term());
            uint64_t doc_freq = iter->doc_freq();
            if (!hidden_docs.empty() && doc_freq != 0) {
                auto hidden_postings = TermIterator::create(
                        io_ctx, false, index_reader, ws_field_name, term_info.get_single_term());
                // Leaping costs a skip per hidden row, scanning a decode per posting: leap only
                // over few hidden rows.
                if (hidden_docs.size() * collection_statistics_detail::kSkipsPerScannedPosting <
                    doc_freq) {
                    doc_freq -= collection_statistics_detail::count_listed_docs(*hidden_postings,
                                                                                hidden_docs);
                } else {
                    if (hidden_doc_bits.empty()) {
                        hidden_doc_bits = collection_statistics_detail::doc_bitset(
                                hidden_docs, static_cast<uint32_t>(index_doc_count));
                    }
                    doc_freq -= collection_statistics_detail::count_listed_docs(*hidden_postings,
                                                                                hidden_doc_bits);
                }
            }
            _term_doc_freqs[ws_field_name][iter->term()] += doc_freq;
        }
    }

    _total_num_docs += total_seg_num_docs;
    _counted_hidden_docs = _counted_hidden_docs || !hidden_docs.empty();

    return Status::OK();
}

void CollectionStatistics::copy_collected_from(const CollectionStatistics& other) {
    _total_num_docs = other._total_num_docs;
    _counted_hidden_docs = other._counted_hidden_docs;
    _total_num_tokens = other._total_num_tokens;
    _term_doc_freqs = other._term_doc_freqs;
    _avg_dl_by_col.clear();
    _idf_by_col_term.clear();
}

TabletCollectionStatistics::TabletCollectionStatistics(std::vector<RowSetSplits> rs_splits)
        : _rs_splits(std::move(rs_splits)) {}

TabletCollectionStatistics::~TabletCollectionStatistics() = default;

Status TabletCollectionStatistics::copy_to(RuntimeState* state,
                                           const TabletSchemaSPtr& tablet_schema,
                                           const VExprContextSPtrs& common_expr_ctxs_push_down,
                                           io::IOContext* io_ctx, const MowHiddenRows* hidden_rows,
                                           CollectionStatistics* statistics) {
    DORIS_CHECK(statistics != nullptr);
    std::lock_guard lock(_mutex);
    if (!_collected) {
        // Every scanner of the tablet reads with the same delete bitmap and version.
        _status = _statistics.collect(state, _rs_splits, tablet_schema, common_expr_ctxs_push_down,
                                      io_ctx, hidden_rows);
        _collected = true;
        // Do not hold the rowsets any longer than needed.
        _rs_splits.clear();
    }
    RETURN_IF_ERROR(_status);
    statistics->copy_collected_from(_statistics);
    return Status::OK();
}

uint64_t CollectionStatistics::get_term_doc_freq_by_col(const std::wstring& lucene_col_name,
                                                        const std::wstring& term) {
    if (!_term_doc_freqs.contains(lucene_col_name)) {
        throw Exception(ErrorCode::INVERTED_INDEX_CLUCENE_ERROR,
                        "Index statistics collection failed: Not such column {}",
                        StringHelper::to_string(lucene_col_name));
    }

    if (!_term_doc_freqs[lucene_col_name].contains(term)) {
        throw Exception(ErrorCode::INVERTED_INDEX_CLUCENE_ERROR,
                        "Index statistics collection failed: Not such term {}",
                        StringHelper::to_string(term));
    }

    return _term_doc_freqs[lucene_col_name][term];
}

uint64_t CollectionStatistics::get_total_term_cnt_by_col(const std::wstring& lucene_col_name) {
    if (!_total_num_tokens.contains(lucene_col_name)) {
        throw Exception(ErrorCode::INVERTED_INDEX_CLUCENE_ERROR,
                        "Index statistics collection failed: Not such column {}",
                        StringHelper::to_string(lucene_col_name));
    }

    return _total_num_tokens[lucene_col_name];
}

uint64_t CollectionStatistics::get_doc_num() const {
    // A tablet whose rows are all hidden has no document a query can match, so its statistics
    // are empty rather than missing.
    if (_total_num_docs == 0 && !_counted_hidden_docs) {
        throw Exception(
                ErrorCode::INVERTED_INDEX_CLUCENE_ERROR,
                "Index statistics collection failed: No data available for SimilarityCollector");
    }

    return _total_num_docs;
}

float CollectionStatistics::get_or_calculate_avg_dl(const std::wstring& lucene_col_name) {
    auto iter = _avg_dl_by_col.find(lucene_col_name);
    if (iter != _avg_dl_by_col.end()) {
        return iter->second;
    }

    const uint64_t total_term_cnt = get_total_term_cnt_by_col(lucene_col_name);
    const uint64_t total_doc_cnt = get_doc_num();
    float avg_dl = total_doc_cnt > 0 ? float((double)total_term_cnt / (double)total_doc_cnt) : 1.0F;
    _avg_dl_by_col[lucene_col_name] = avg_dl;
    return avg_dl;
}

float CollectionStatistics::get_or_calculate_idf(const std::wstring& lucene_col_name,
                                                 const std::wstring& term) {
    auto iter = _idf_by_col_term.find(lucene_col_name);
    if (iter != _idf_by_col_term.end()) {
        auto term_iter = iter->second.find(term);
        if (term_iter != iter->second.end()) {
            return term_iter->second;
        }
    }

    const uint64_t doc_num = get_doc_num();
    const uint64_t doc_freq = get_term_doc_freq_by_col(lucene_col_name, term);
    auto idf = (float)std::log(1 + ((double)doc_num - (double)doc_freq + (double)0.5) /
                                           ((double)doc_freq + (double)0.5));
    _idf_by_col_term[lucene_col_name][term] = idf;
    return idf;
}

#include "common/compile_check_end.h"
} // namespace doris
