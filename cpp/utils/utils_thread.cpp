//
// Created by dell on 2023/8/14.
//
#include <torch/torch.h>
#include <chrono>
#include <cstdio>
#include <thread>
#include <future>
#include <cstring>
#include <unordered_map>
#include <spdlog/spdlog.h>
#include <set>
#include "../DataLoader/Reference_Reader.h"
#include "../3rdparty/threadpool/threadpool.h"
#include "utils_thread.h"
#include "../3rdparty/cnpy/cnpy.h"

void Yao::get_hc_features_subthread(Yao::Pod5Data &p5,
                                    std::vector<Yao::SamRead *> inputs,
                                    std::map<std::string, std::string> & reformat_chr,
//                                    fs::path & writefile,
                                    std::vector<uint8_t> & total_site_info_lists,
                                    std::vector<float> & total_feature_lists,
                                    std::set<std::string> &pos_hc_sites,
                                    std::set<std::string> &neg_hc_sites,
                                    std::mutex &mtx,
                                    std::condition_variable &cv,
                                    int32_t kmer_size,
                                    int32_t mapq_thresh_hold,
                                    float coverage_thresh_hold,
                                    float identity_thresh_hold,
                                    std::set<std::string> & motifset,
                                    size_t loc_in_motif,
                                    std::atomic<int64_t> &total_cnt, 
                                    std::atomic<int64_t> &thread_cnt) {

    thread_cnt ++;

    using namespace torch::indexing;
    std::vector<uint8_t> site_info_lists;
    std::vector<float> feature_lists;
    const auto base2code_dna = Yao::get_base2code_dna();
    int cnt = 0;
    std::stringstream  ss;
    auto convert_num2str = [&](int32_t i) -> std::string{
        ss.str("");
        ss.str("");
        ss << std::setw(10) << std::setfill('0') << i;
        std::string s = ss.str();
        assert(s.length() == size_t(10));
        return s;
    };
    for (auto & sam_ptr : inputs){
        auto coverage = Yao::compute_coverage(sam_ptr->cigar_pair, sam_ptr->query_alignment_length);
        auto identity = Yao::compute_pct_identity(sam_ptr->cigar_pair, sam_ptr->NM);
        if (sam_ptr->mapping_quality < mapq_thresh_hold || coverage < coverage_thresh_hold || identity < identity_thresh_hold) {
            delete sam_ptr; sam_ptr = nullptr;
            continue;
        }
        if (sam_ptr->stride == -1 || sam_ptr->movetable.empty()) {
            delete sam_ptr; sam_ptr = nullptr;
            continue;
        }
        std::string read_id = sam_ptr->query_name;
        auto normed_sig = p5.get_normalized_signal_by_read_id(read_id);
        auto trimed_sig = normed_sig.index({Slice{sam_ptr->trimed_start, None}});

        std::vector<at::Tensor> signal_group;
        Yao::group_signal_by_movetable(signal_group,
                                       trimed_sig,
                                       sam_ptr->movetable,
                                       sam_ptr->stride);
        std::string ref_seq = sam_ptr->reference_seq;
        std::string query_seq = sam_ptr->query_sequence;
        int32_t read_start = sam_ptr->query_alignment_start;
        at::Tensor query_qualities = Yao::get_query_qualities(sam_ptr->query_qualities,
                                                              sam_ptr->is_forward);
        if (!sam_ptr->is_forward) {
            ref_seq = Yao::get_complement_seq(ref_seq, "DNA");
            query_seq = Yao::get_complement_seq(query_seq, "DNA");
            read_start = query_seq.length() - sam_ptr->query_alignment_end;
        }
        std::string strand_code = sam_ptr->is_forward ? "+" : "-";
        std::vector<int32_t> r_to_q_poss;
        Yao::parse_cigar(sam_ptr->cigar_pair,
                         r_to_q_poss,
                         sam_ptr->is_forward,
                         sam_ptr->reference_length);
        if (kmer_size % 2 == 0) {
            spdlog::error("Kmer size must be odd!");
        }
        int32_t  num_bases = (kmer_size - 1) / 2;

        std::vector<int32_t> refloc_of_methysite;
        Yao::get_refloc_of_methysite_in_motif(refloc_of_methysite,
                                              ref_seq,
                                              motifset,
                                              loc_in_motif);
        std::vector<int32_t> ref_readlocs(ref_seq.length(), 0);
        std::vector<at::Tensor> ref_signal_grp(ref_seq.length(), at::Tensor());
        at::Tensor ref_base_qual = torch::empty(ref_seq.length(), torch::kFloat32);
        for (int32_t ref_pos = 0; ref_pos < (int32_t)r_to_q_poss.size() - 1; ref_pos++) {
            ref_readlocs[ref_pos] = r_to_q_poss[ref_pos] + read_start;
            ref_signal_grp[ref_pos] = signal_group[r_to_q_poss[ref_pos] + read_start];
            ref_base_qual[ref_pos] = query_qualities[r_to_q_poss[ref_pos] + read_start];
        }
        for (const auto &off_loc : refloc_of_methysite) {
            if (num_bases <= off_loc && (off_loc + num_bases) < (int32_t)ref_seq.length() ) {
                int32_t abs_loc = sam_ptr->reference_start + off_loc;
                if (!sam_ptr->is_forward) {
                    abs_loc = sam_ptr->reference_end - 1 - off_loc;
                }
                std::string site_key = sam_ptr->reference_name + \
                    "#" + std::to_string(abs_loc);
                if (pos_hc_sites.find(site_key) == pos_hc_sites.end() && neg_hc_sites.find(site_key) == neg_hc_sites.end()) {
                    continue;
                }

                std::string k_mer = ref_seq.substr(off_loc - num_bases, kmer_size);
                std::vector<int64_t> kmer_v(kmer_size);
                for (size_t i = 0; i < k_mer.length(); i++) {
                    kmer_v[i] = base2code_dna.at(k_mer[i]);
                }
                at::Tensor kmer_t = torch::from_blob(kmer_v.data(), kmer_v.size(), torch::kInt64);
                // kmer_t += 4 * strand_code;
                std::vector<at::Tensor> k_signals(kmer_size);
                for (int32_t pos = off_loc - num_bases; pos < off_loc + num_bases + 1; pos++) {
                    k_signals[pos - off_loc + num_bases] = ref_signal_grp[pos];
                }
                at::Tensor k_seq_qual = ref_base_qual.index({Slice(off_loc - num_bases, off_loc + num_bases + 1)});
                std::vector<float> signal_lens(kmer_size), signal_means(kmer_size), signal_stds(kmer_size);

                for (int32_t i = 0; i < kmer_size; i++) {
                    signal_lens[i] = (float)k_signals[i].size(0);
                    signal_means[i] = torch::mean(k_signals[i]).item<float>();
                    signal_stds[i] = torch::std(k_signals[i], 0, false).item<float>();
                }
                at::Tensor signal_lens_t = torch::from_blob(signal_lens.data(),
                                                            signal_lens.size(),
                                                            torch::kFloat32);

                at::Tensor signal_means_t = torch::from_blob(signal_means.data(),
                                                             signal_means.size(),
                                                             torch::kFloat32);
                at::Tensor signal_stds_t = torch::from_blob(signal_stds.data(),
                                                            signal_stds.size(),
                                                            torch::kFloat32);
                at::Tensor k_signals_rect = Yao::get_signals_rect(k_signals, 15);
                signal_means_t = signal_means_t.reshape({kmer_size, 1});
                signal_stds_t = signal_stds_t.reshape({kmer_size, 1});
                signal_lens_t = signal_lens_t.reshape({kmer_size, 1});
                k_seq_qual = k_seq_qual.reshape({kmer_size, 1});
                at::Tensor feature = torch::concat({signal_means_t,
                                                    signal_stds_t,
                                                    signal_lens_t,
                                                    k_seq_qual,
                                                    k_signals_rect}, 1).reshape({-1});
                feature = torch::concat({kmer_t, feature, }, 0);
                feature = feature.round(6);
                feature = feature.contiguous();
                std::vector<float> feature_v(feature.data_ptr<float>(),
                                             feature.data_ptr<float>() + feature.numel());
                std::string site_info = sam_ptr->query_name + "\t" + \
                    convert_num2str(sam_ptr->reference_start) + "\t" +  \
                    convert_num2str(sam_ptr->reference_end) + "\t" + \
                    reformat_chr.at(sam_ptr->reference_name) + \
                    "\t" + convert_num2str(abs_loc) + "\t" + strand_code;
                site_info_lists.insert(site_info_lists.end(), site_info.begin(), site_info.end());

                if (pos_hc_sites.find(site_key) != pos_hc_sites.end()) {
                    feature_v.push_back(1.0);
                }else {
                    feature_v.push_back(0.0);
                }
                cnt += 1;
                feature_lists.insert(feature_lists.end(),
                                     feature_v.begin(),
                                     feature_v.end());
            }
        }
        delete sam_ptr;
        sam_ptr = nullptr;
    }
    for (auto &ptr : inputs) {
        if (ptr != nullptr) {
            delete ptr;
            ptr = nullptr;
        }
    }
    inputs.clear();
    {
        std::unique_lock<std::mutex> lock(mtx);
        total_site_info_lists.insert(total_site_info_lists.end(),
                                     site_info_lists.begin(),
                                     site_info_lists.end());
        total_feature_lists.insert(total_feature_lists.end(),
                                   feature_lists.begin(),
                                   feature_lists.end());
//        if (fs::exists(writefile)) {
//            cnpy::npy_save(writefile.string(),
//                           &feature_lists[0],
//                           {feature_lists.size() / len, len},
//                           "a");
//        }
//        else {
//            cnpy::npy_save(writefile.string(),
//                           &feature_lists[0],
//                           {feature_lists.size() / len, len},
//                           "w");
//        }

    }
    cv.notify_all();
    size_t len = kmer_size * 20 + 1;
    total_cnt += feature_lists.size() / len;
    thread_cnt--;
}

void Yao::get_hc_features(Yao::Pod5Data p5,
                       std::vector<Yao::SamRead*> inputs,
                       std::map<std::string, std::string> & reformat_chr,
                       fs::path writefile,
                       std::set<std::string> & pos_hc_sites,
                       std::set<std::string> & neg_hc_sites,
                       int32_t kmer_size,
                       int32_t mapq_thresh_hold,
                       float coverage_thresh_hold,
                       float identity_thresh_hold,
                       std::set<std::string> & motifset,
                       size_t loc_in_motif, 
                       std::atomic<int64_t> &thread_cnt) {
    auto st = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> total_site_info_lists;
    std::vector<float> total_feature_lists;
    int32_t stride = (inputs.size() + 2) / 2;
    std::atomic<int64_t> total_cnt(0);
    std::mutex mtx;
    std::condition_variable cv;
    std::vector<std::thread> workers;
    for (size_t i = 0; i < inputs.size(); i += stride) {
        auto st = inputs.begin() + i;
        auto ed = inputs.begin() + std::min(inputs.size(), i + stride);
        std::vector<Yao::SamRead*> input_s(st, ed);
        workers.emplace_back(Yao::get_hc_features_subthread,
                std::ref(p5),
                input_s,
                std::ref(reformat_chr),
                std::ref(total_site_info_lists),
                std::ref(total_feature_lists),
                std::ref(pos_hc_sites),
                std::ref(neg_hc_sites),
                std::ref(mtx),
                std::ref(cv),
                kmer_size,
                mapq_thresh_hold,
                coverage_thresh_hold,
                identity_thresh_hold,
                std::ref(motifset),
                loc_in_motif,
                std::ref(total_cnt), 
                std::ref(thread_cnt)
                );
    }
    for (std::thread &worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    inputs.clear();
    p5.release();
    size_t feature_len = kmer_size * 20 + 1;
    size_t sample_len = total_feature_lists.size() / feature_len;
    cnpy::npz_save(writefile,
                   "site_infos",
                   &total_site_info_lists[0],
                   {sample_len, total_site_info_lists.size() / sample_len},
                   "w");
    cnpy::npz_save(writefile,
                   "features",
                   &total_feature_lists[0],
                   {sample_len, feature_len},
                   "a");
    auto ed = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(ed - st);
    spdlog::info("Extract features for {} finished, extracted {} features, cost {} seconds",
                 p5.get_filename(), total_cnt, (float)duration.count() / 1000.);
}

void Yao::get_feature_for_model_subthread(Yao::Pod5Data &p5,
                                          std::vector<Yao::SamRead *> inputs,
                                          std::queue<std::tuple<std::vector<std::string>,\
                                                     std::vector<std::string>, \
                                                     at::Tensor, at::Tensor>> &dataQueue,
                                          std::mutex &mtx1,
                                          std::condition_variable &cv1,
                                          int32_t batch_size,
                                          int32_t kmer_size,
                                          int32_t mapq_thresh_hold,
                                          float coverage_thresh_hold,
                                          float identity_thresh_hold,
                                          std::set<std::string> & motifset,
                                          size_t loc_in_motif,
                                          std::atomic<int32_t> &total_count,
                                          std::atomic<int64_t> &thread_cnt) {
    thread_cnt++;

    using namespace torch::indexing;
    const auto base2code_dna = Yao::get_base2code_dna();
    int cnt = 0, thread_total_cnt = 0;
    std::vector<std::string> site_key_batch(batch_size);
    std::vector<std::string> site_info_batch(batch_size);
    at::Tensor kmer_batch = torch::empty({batch_size, kmer_size}, torch::kInt64);
    at::Tensor signal_batch = torch::empty({batch_size, 1, kmer_size, 19});
    for (auto & sam_ptr : inputs){
        auto coverage = Yao::compute_coverage(sam_ptr->cigar_pair, sam_ptr->query_alignment_length);
        auto identity = Yao::compute_pct_identity(sam_ptr->cigar_pair, sam_ptr->NM);
        if (sam_ptr->mapping_quality < mapq_thresh_hold || coverage < coverage_thresh_hold || identity < identity_thresh_hold) {
            delete sam_ptr; sam_ptr = nullptr;
            continue;
        }
        if (sam_ptr->stride == -1 || sam_ptr->movetable.size() == 0) {
            delete sam_ptr; sam_ptr = nullptr;
            continue;
        }

        std::string read_id = sam_ptr->query_name;
        auto normed_sig = p5.get_normalized_signal_by_read_id(read_id);
        auto trimed_sig = normed_sig.index({Slice{sam_ptr->trimed_start, None}});

        std::vector<at::Tensor> signal_group;
        Yao::group_signal_by_movetable(signal_group,
                                       trimed_sig,
                                       sam_ptr->movetable,
                                       sam_ptr->stride);
        std::string ref_seq = sam_ptr->reference_seq;
        std::string query_seq = sam_ptr->query_sequence;
        int32_t read_start = sam_ptr->query_alignment_start;
        at::Tensor query_qualities = Yao::get_query_qualities(sam_ptr->query_qualities,
                                                              sam_ptr->is_forward);
        if (!sam_ptr->is_forward) {
            ref_seq = Yao::get_complement_seq(ref_seq, "DNA");
            query_seq = Yao::get_complement_seq(query_seq, "DNA");
            read_start = query_seq.length() - sam_ptr->query_alignment_end;
        }

        std::string strand_code = "+";
        if (!sam_ptr->is_forward) {
            strand_code = "-";
        }

        std::vector<int32_t> r_to_q_poss;
        Yao::parse_cigar(sam_ptr->cigar_pair,
                         r_to_q_poss,
                         sam_ptr->is_forward,
                         sam_ptr->reference_length);
        if (kmer_size % 2 == 0) {
            spdlog::error("Kmer size must be odd!");
        }
        int32_t  num_bases = (kmer_size - 1) / 2;

        std::vector<int32_t> refloc_of_methysite;
        Yao::get_refloc_of_methysite_in_motif(refloc_of_methysite,
                                              ref_seq,
                                              motifset,
                                              loc_in_motif);
        std::vector<int32_t> ref_readlocs(ref_seq.length(), 0);
        std::vector<at::Tensor> ref_signal_grp(ref_seq.length(), at::Tensor());
        at::Tensor ref_base_qual = torch::empty(ref_seq.length(), torch::kFloat32);
        for (int32_t ref_pos = 0; ref_pos < (int32_t)r_to_q_poss.size() - 1; ref_pos++) {
            ref_readlocs[ref_pos] = r_to_q_poss[ref_pos] + read_start;
            ref_signal_grp[ref_pos] = signal_group[r_to_q_poss[ref_pos] + read_start];
            ref_base_qual[ref_pos] = query_qualities[r_to_q_poss[ref_pos] + read_start];
        }
        for (const auto &off_loc : refloc_of_methysite) {
            if (num_bases <= off_loc && (off_loc + num_bases) < (int32_t)ref_seq.length() ) {
                int32_t abs_loc = sam_ptr->reference_start + off_loc;
                if (!sam_ptr->is_forward) {
                    abs_loc = sam_ptr->reference_end - 1 - off_loc;
                }
                std::string k_mer = ref_seq.substr(off_loc - num_bases, kmer_size);
                std::vector<int64_t> kmer_v(kmer_size);
                for (size_t i = 0; i < k_mer.length(); i++) {
                    kmer_v[i] = base2code_dna.at(k_mer[i]);
                }
                at::Tensor kmer_t = torch::from_blob(kmer_v.data(), kmer_v.size(), torch::kInt64);
                // if (sam_ptr->is_forward) kmer_t += 4;
                std::vector<at::Tensor> k_signals(kmer_size);
                for (int32_t pos = off_loc - num_bases; pos < off_loc + num_bases + 1; pos++) {
                    k_signals[pos - off_loc + num_bases] = ref_signal_grp[pos];
                }
                at::Tensor k_seq_qual = ref_base_qual.index({Slice(off_loc - num_bases, off_loc + num_bases + 1)});
                std::vector<float> signal_lens(kmer_size), signal_means(kmer_size), signal_stds(kmer_size);

                for (int32_t i = 0; i < kmer_size; i++) {
                    signal_lens[i] = (float)k_signals[i].size(0);
                    signal_means[i] = torch::mean(k_signals[i]).item<float>();
                    signal_stds[i] = torch::std(k_signals[i], 0, false).item<float>();
                }
                at::Tensor signal_lens_t = torch::from_blob(signal_lens.data(),
                                                            signal_lens.size(),
                                                            torch::kFloat32);
                // doing median absolute difference for signal_len
                // todo: may try to drop this in the furture
//                auto center = torch::median(signal_lens_t);
//                float c = 0.674489;
//                auto mad = torch::median(torch::abs(signal_lens_t - center)) / c;
//                if (mad.item<float>() != 0.0)
//                    signal_lens_t = (signal_lens_t - torch::median(signal_lens_t)) / mad;
//                signal_lens_t = signal_lens_t.to(torch::kFloat32);

                at::Tensor signal_means_t = torch::from_blob(signal_means.data(),
                                                             signal_means.size(),
                                                             torch::kFloat32);
                at::Tensor signal_stds_t = torch::from_blob(signal_stds.data(),
                                                            signal_stds.size(),
                                                            torch::kFloat32);
                at::Tensor k_signals_rect = Yao::get_signals_rect(k_signals, 15);
                signal_means_t = signal_means_t.reshape({(long)kmer_size, 1});
                signal_stds_t = signal_stds_t.reshape({(long)kmer_size, 1});
                signal_lens_t = signal_lens_t.reshape({(long)kmer_size, 1});
                k_seq_qual = k_seq_qual.reshape({(long)kmer_size, 1});
                at::Tensor signal = torch::concat({signal_means_t,
                                                   signal_stds_t,
                                                   signal_lens_t,
                                                   k_seq_qual,
                                                   k_signals_rect}, 1)
                        .reshape({ 1, (long)kmer_size, 19});

                // get site key (chr, pos in strand, strand)
                // and detailed site info (read_id, ref_st, ref_ed, chr, pos_in_strand, strand)
                std::string site_key = sam_ptr->reference_name + \
                    "\t" + std::to_string(abs_loc) + "\t" + strand_code;
                std::string site_info = sam_ptr->query_name + "\t" + \
                    std::to_string(sam_ptr->reference_start) + "\t" +  \
                    std::to_string(sam_ptr->reference_end) + "\t" + \
                    sam_ptr->reference_name + \
                    "\t" + std::to_string(abs_loc) + "\t" + strand_code;

                site_key_batch[cnt % batch_size] = site_key;
                site_info_batch[cnt  % batch_size] = site_info;
                kmer_batch[cnt % batch_size] = kmer_t;
                signal_batch[cnt % batch_size] = signal;

                cnt += 1; thread_total_cnt += 1;
                if (cnt == batch_size) {
                    cnt = 0;
                    {
                        std::unique_lock<std::mutex> lock(mtx1);
                        kmer_batch = kmer_batch.to(torch::kLong);
                        dataQueue.push({site_key_batch,
                                        site_info_batch,
                                        kmer_batch.clone(),
                                        signal_batch.clone()});
                    }
                    cv1.notify_all();
                }
            }
        }
        delete sam_ptr;
        sam_ptr = nullptr;
    }
    if (cnt > 0) {
        auto st1 = site_key_batch.begin();
        auto ed1 = site_key_batch.begin() + cnt;
        auto st2 = site_info_batch.begin();
        auto ed2 = site_info_batch.begin() + cnt;

        std::vector<std::string> site_key_batch_ch(st1, ed1);
        std::vector<std::string> site_info_batch_ch(st2, ed2);

        {
            std::unique_lock<std::mutex> lock(mtx1);
            kmer_batch = kmer_batch.to(torch::kLong);
            dataQueue.push({site_key_batch_ch,
                            site_info_batch_ch,
                            kmer_batch.index({Slice(0, cnt)}).clone(),
                            signal_batch.index({Slice(0, cnt)}).clone()});
        }

        cv1.notify_all();
    }
    for (auto &ptr : inputs) {
        if (ptr != nullptr) {
            delete ptr;
            ptr = nullptr;
        }
    }
    inputs.clear();
    total_count += thread_total_cnt;
    thread_cnt--;
}


void Yao::get_feature_for_model(Yao::Pod5Data p5,
                                // p5 pass by value, because it is created as temperate variable with pointer to allocated memory
                                std::vector<Yao::SamRead *> inputs,
                                std::queue<std::tuple<std::vector<std::string>,\
                                         std::vector<std::string>, \
                                         at::Tensor, at::Tensor>> &dataQueue,
                                std::mutex &mtx1,
                                std::condition_variable &cv1,
                                int32_t batch_size,
                                int32_t kmer_size,
                                int32_t mapq_thresh_hold,
                                float coverage_thresh_hold,
                                float identity_thresh_hold,
                                std::set<std::string>  & motifset,
                                size_t loc_in_motif,
                                std::atomic<int64_t>& thread_cnt) {

    auto st = std::chrono::high_resolution_clock::now();
    auto thread_id = std::this_thread::get_id(); // convert thread id to string
    std::stringstream sstream;
    sstream << thread_id;
    std::string thread_str = sstream.str();
//    spdlog::info("Thread-{} start to extract features for {}, found {} reads",
//                 thread_str, p5.get_filename(), inputs.size());
    std::vector<std::thread> workers;
    size_t stride = (inputs.size() + 4) / 4;
    std::atomic<int32_t> total_cnt = 0;

    for (size_t i = 0; i < inputs.size(); i += stride) {
        auto st = inputs.begin() + i;
        auto ed = inputs.begin() + std::min(i + stride, inputs.size());
        std::vector<Yao::SamRead*> input_s(st, ed);
        workers.emplace_back(Yao::get_feature_for_model_subthread,
                                std::ref(p5),
                                std::move(input_s),
                                std::ref(dataQueue),
                                std::ref(mtx1),
                                std::ref(cv1),
                                batch_size,
                                kmer_size,
                                mapq_thresh_hold,
                                coverage_thresh_hold,
                                identity_thresh_hold,
                                std::ref(motifset),
                                loc_in_motif,
                                std::ref(total_cnt),
                                std::ref(thread_cnt));
    }
    for (std::thread & worker : workers){
        if (worker.joinable()) {
            worker.join();
        }
    }

    inputs.clear();
    p5.release();

    auto ed = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(ed - st);
    spdlog::info("Extract features for {} finished, extracted {} features to module, cost {} seconds", p5.get_filename(),
                 total_cnt, (float)duration.count() / 1000.);
}

void Yao::get_feature_for_model_with_thread_pool(int32_t num_workers,
                                                 fs::path &pod5_dir,
                                                 fs::path &bam_path,
                                                 Yao::Reference_Reader & ref,
                                                 std::queue<std::tuple<std::vector<std::string>,\
                                                         std::vector<std::string>, \
                                                         at::Tensor, at::Tensor>> &dataQueue,
                                                 std::mutex &mtx1,
                                                 std::condition_variable &cv1,
                                                 int32_t batch_size,
                                                 int32_t kmer_size,
                                                 int32_t mapq_thresh_hold,
                                                 float coverage_thresh_hold,
                                                 float identity_thresh_hold,
                                                 std::set<std::string> & motifset,
                                                 size_t loc_in_motif) {
    spdlog::info("Start to get feature for call modification");
    const auto filename_to_path = Yao::get_filename_to_path(pod5_dir);
    std::atomic<int64_t> thread_cnt(0);
    {
        ThreadPool pool(num_workers * 2);
        const size_t buffer_size = 1024 * 1024 * 50;
        char * buffer = new char [buffer_size];
        std::string sam_str;
        std::string cmd = "samtools view -@ 12 " + bam_path.string();
        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);

        if (!pipe)
        {
            throw std::runtime_error("popen() failed!");
        }
        int32_t file_cnt = 0;
        std::string file_name_hold = "";
        std::vector<Yao::SamRead *> inputs;
        while (fgets(buffer, buffer_size, pipe.get()) != nullptr) {

            sam_str = buffer;
            Yao::SamRead *sam_ptr = new Yao::SamRead(sam_str);
            if (file_name_hold != sam_ptr->file_name && !file_name_hold.empty()) {
                fs::path p5_file;
                try {
                    p5_file = filename_to_path.at(file_name_hold);
                    Yao::Pod5Data p5(p5_file);

                    while (thread_cnt >= num_workers * 4) {
                        std::this_thread::sleep_for(std::chrono::seconds(1));
                    }

                    // todo reduce memory cost in extract feature thread pool
                    pool.enqueue(Yao::get_feature_for_model,
                                 std::move(p5),
                                 inputs,
                                 std::ref(dataQueue),
                                 std::ref(mtx1),
                                 std::ref(cv1),
                                 batch_size,
                                 kmer_size,
                                 mapq_thresh_hold,
                                 coverage_thresh_hold,
                                 identity_thresh_hold,
                                 std::ref(motifset),
                                 loc_in_motif,
                                 std::ref(thread_cnt));
                    inputs.clear();
                    file_cnt++;
                    spdlog::info("File {} enter to"\
                         " thread pool, progress [{}/{}], current working thread: {}",
                                 p5.get_filename(), file_cnt, filename_to_path.size(), thread_cnt);
                }catch (...) {
                    spdlog::error("Couldn't find file: {}", file_name_hold);
                    file_name_hold = sam_ptr->file_name;
                    // release allocated memory
                    for (auto &ptr : inputs) {
                        delete ptr;
                    }
                    inputs.clear();
                }
            }

            file_name_hold = sam_ptr->file_name;
            if (!sam_ptr->is_mapped) {
                delete sam_ptr;
                sam_ptr = nullptr;
                continue;
            }
            int32_t st = sam_ptr->reference_start;
            int32_t ed = sam_ptr->reference_end;
            std::string chr_key = sam_ptr->reference_name;
            sam_ptr->reference_seq = ref.get_reference_seq(chr_key, sam_ptr->is_forward, st, ed);
            if (sam_ptr->reference_seq.length() == 0) {
                delete sam_ptr;
                sam_ptr = nullptr;
                continue;
            }
            inputs.push_back(sam_ptr);
        }
        if (!inputs.empty()) {
            fs::path p5_file;
            try {
                p5_file = filename_to_path.at(file_name_hold);
                Yao::Pod5Data p5(p5_file);
                pool.enqueue(Yao::get_feature_for_model,
                             p5,
                             inputs,
                             std::ref(dataQueue),
                             std::ref(mtx1),
                             std::ref(cv1),
                             batch_size,
                             kmer_size,
                             mapq_thresh_hold,
                             coverage_thresh_hold,
                             identity_thresh_hold,
                             std::ref(motifset),
                             loc_in_motif,
                             std::ref(thread_cnt));
                inputs.clear();
                file_cnt ++;
                spdlog::info("File {} enter to"\
                         " thread pool, progress [{}/{}]",
                             p5.get_filename(), file_cnt, filename_to_path.size());
            }
            catch (...) {
                spdlog::error("Couldn't find file: {}", file_name_hold);
                for (auto &ptr : inputs) {
                    delete ptr;
                }
                inputs.clear();
            }
        }
        delete [] buffer;
        // wait for thread pool to finish, feature extraction and deconstruct
        // thread pool deconstruction will join all threads to work
    }
    // push fake feature as ending
    {
        std::unique_lock<std::mutex> lock(mtx1);
        dataQueue.push({{}, {}, at::Tensor(), at::Tensor()});
    }
    cv1.notify_all();
}

void
Yao::Model_Inference(torch::jit::Module &module,
                     std::queue<std::tuple<std::vector<std::string>,\
                                                         std::vector<std::string>, \
                                                         at::Tensor, at::Tensor>> &dataQueue,
                     std::queue<std::vector<std::string>> & site_key_Queue,
                     std::queue<std::vector<std::string>> & site_info_Queue,
                     std::queue<std::vector<int64_t>> & pred_Queue,
                     std::queue<std::vector<float>> & p_rate_Queue,
                     std::mutex &mtx1,
                     std::condition_variable &cv1,
                     std::mutex &mtx2,
                     std::condition_variable &cv2,
                     size_t batch_size) {
    using namespace torch::indexing;
    std::vector<torch::jit::IValue> inputs;
    int64_t cnt = 0;
    while (true) {
        std::unique_lock<std::mutex> lock(mtx1);
        cv1.wait(lock, [&dataQueue]{
            return !dataQueue.empty();
        }); // Wait for data to process

        auto [site_key, site_info, kmer, signal] = dataQueue.front(); dataQueue.pop();
        lock.unlock();
        cv1.notify_all();
        if (kmer.size(0) == 0) {
            break;
        }
        kmer = kmer.to(torch::kLong);
        signal = signal.to(torch::kFloat32);
        inputs.push_back(kmer.to(torch::kCUDA));
        inputs.push_back(signal.to(torch::kCUDA));

        auto result = module.forward(inputs);
        at::Tensor logits = result.toTuple()->elements()[1].toTensor();
        at::Tensor pred = logits.argmax(1);

        pred = pred.to(torch::kCPU);
        pred = pred.contiguous();
        std::vector<int64_t> pred_v(pred.data_ptr<int64_t>(),
                pred.data_ptr<int64_t>() + pred.numel());

        logits = logits.to(torch::kCPU);
        logits = logits.index({Slice{}, Slice{1,None}});
        logits = logits.contiguous(); // make sure Tensor is stored at a contninuous space before convert to a vector
        std::vector<float> p_rate_v(logits.data_ptr<float>(),
                logits.data_ptr<float>() + logits.numel());
        inputs.clear();
//        spdlog::info("p_rate_v: {}, kmer: {}", p_rate_v.size(), kmer.size(0));
        {
            std::unique_lock<std::mutex> lock2(mtx2);
            site_key_Queue.push(site_key);
            site_info_Queue.push(site_info);
            pred_Queue.push(pred_v);
            p_rate_Queue.push(p_rate_v);
        }
        cv2.notify_one();
        cnt += 1;
    }
    std::unique_lock<std::mutex> lock2(mtx2);
    site_key_Queue.push({});
    site_info_Queue.push({});
    pred_Queue.push({});
    p_rate_Queue.push({});// push vector of size 0 as signal of model_inference_thread finished
    lock2.unlock();
    cv2.notify_one();
    auto thread_id = std::this_thread::get_id(); // convert thread id to string
    std::stringstream sstream;
    sstream << thread_id;
    std::string thread_str = sstream.str();
    spdlog::info("Model inference thread-{} finished, processed {} batch", thread_str, cnt);
}


void Yao::count_modification_thread(std::map<std::string, std::vector<float>> &site_dict,
                                    std::queue<std::vector<std::string>> &site_key_Queue,
                                    std::queue<std::vector<std::string>> &site_info_Queue,
                                    std::queue<std::vector<int64_t>> &pred_Queue,
                                    std::queue<std::vector<float>> & p_rate_Queue,
                                    fs::path &write_file2,
                                    std::mutex &mtx2,
                                    std::condition_variable &cv2) {
    std::ofstream out;
    out.open(write_file2);
    out << "read_id\treference_start\treference_end\tchromosome\tpos_in_strand\tstrand\tmethylation_rate\n";
    while (true) {
        std::unique_lock<std::mutex> lock(mtx2);
        cv2.wait(lock, [&site_key_Queue]{
            return !site_key_Queue.empty();
        });
        auto site_key = site_key_Queue.front(); site_key_Queue.pop();
        auto site_info = site_info_Queue.front(); site_info_Queue.pop();
        auto pred_v = pred_Queue.front(); pred_Queue.pop();
        auto p_rate = p_rate_Queue.front(); p_rate_Queue.pop();
        lock.unlock();
        // break loop when find empty() signal
        if (site_key.empty()) break;
        for (size_t j = 0; j < site_key.size(); j++) {
            out << site_info[j] << "\t" << std::to_string(p_rate[j]) << std::endl;
            if (site_dict.find(site_key[j]) == site_dict.end()) {
                site_dict[site_key[j]] = {0., 0., 0.};
            }
            site_dict[site_key[j]][pred_v[j]] += 1;
            site_dict[site_key[j]][2] += 1;
        }
    }
    out.close();
}

