//
// Created by dell on 2023/9/21.
//

#include "Data_Filter.h"

#include <torch/script.h>
#include <spdlog/spdlog.h>
#include <torch/torch.h>
#include <chrono>
#include <cstdio>
#include <thread>
#include <future>
#include <cstring>
#include <boost/iostreams/categories.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/iostreams/stream.hpp>
#include <spdlog/spdlog.h>
#include <set>

#include "../DataLoader/Reference_Reader.h"
#include "../3rdparty/threadpool/threadpool.h"
#include "../3rdparty/cnpy/cnpy.h"

Yao::Data_Filter::Data_Filter(size_t batch_size_,
                              size_t kmer_size_,
                              fs::path reference_path_,
                              std::string ref_type,
                              fs::path module_path):
                              reference_path(reference_path_),
                              ref(std::move(Yao::Reference_Reader(reference_path_, ref_type))),
                              batch_size(batch_size_),
                              kmer_size(kmer_size_){

    try {
        module = torch::jit::load(module_path);
        spdlog::info("Successfully load module!");
    }
    catch (const c10::Error& e) {
        spdlog::error("Error loading the module");
    }

    module.to(torch::kCUDA);

    // module test
    try {
        at::Tensor kmer = torch::randint(0,8,{512, 51}, torch::kLong);
        at::Tensor signal = torch::rand({512,1,51, 19}, torch::kFloat32);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(kmer.to(torch::kCUDA));
        inputs.push_back(signal.to(torch::kCUDA));
        auto result = module.forward(inputs);
        spdlog::info("module test successfullt!");
    }
    catch (const c10::Error &e) {
        spdlog::error("error loading the module");
        std::cout << e.what() << std::endl;
    }

    // default parameter to filter reads
    mapq_thresh_hold = 20;
    coverage_thresh_hold = 0.8;
    identity_thresh_hold = 0.8;
}

void Yao::Data_Filter::filter_data(fs::path &pod5_dir,
                                   fs::path &bam_path,
                                   fs::path &write_dir,
                                   std::set<std::string> & sites,
                                   size_t &num_workers,
                                   std::set<std::string> &motifset,
                                   size_t &loc_in_motif) {
    auto start = std::chrono::high_resolution_clock::now();
    std::thread thread1(Data_Filter_get_features_for_model_threadpool,
                        num_workers,
                        std::ref(pod5_dir),
                        std::ref(bam_path),
                        std::ref(ref),
                        std::ref(dataQueue),
                        std::ref(sites),
                        std::ref(mtx1),
                        std::ref(cv1),
                        batch_size,
                        kmer_size,
                        mapq_thresh_hold,
                        coverage_thresh_hold,
                        identity_thresh_hold,
                        std::ref(motifset),
                        loc_in_motif);
    std::thread thread2(Data_Filter_Model_Inference,
                        std::ref(module),
                        std::ref(dataQueue),
                        std::ref(data_queue),
                        std::ref(mtx1),
                        std::ref(cv1),
                        std::ref(mtx2),
                        std::ref(cv2),
                        batch_size);
    std::thread thread3(Data_Filter_write_data,
                        std::ref(write_dir),
                        std::ref(data_queue),
                        std::ref(mtx2),
                        std::ref(cv2));

    thread1.join();
    thread2.join();
    thread3.join();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

    spdlog::info("Total time taken: {} seconds", duration.count());
}

void Data_Filter_get_features_for_model_subthread(Yao::Pod5Data &p5,
                                                  std::vector<Yao::SamRead *> inputs,
                                                  std::queue<std::tuple<std::vector<std::string>,
                                                            std::vector<std::string>,
                                                            at::Tensor, at::Tensor>> &dataQueue,
                                                  std::set<std::string> &sites,
                                                  std::mutex &mtx1,
                                                  std::condition_variable &cv1,
                                                  int32_t batch_size,
                                                  int32_t kmer_size,
                                                  int32_t mapq_thresh_hold,
                                                  float coverage_thresh_hold,
                                                  float identity_thresh_hold,
                                                  std::set<std::string> &motifset,
                                                  size_t loc_in_motif,
                                                  std::atomic<int32_t> &total_cnt,
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

                std::string site = sam_ptr->reference_name + "#" + std::to_string(abs_loc);
                if (sites.find(site) == sites.end()) {
                    continue;
                }
                std::string k_mer = ref_seq.substr(off_loc - num_bases, kmer_size);
                std::vector<int64_t> kmer_v(kmer_size);
                for (size_t i = 0; i < k_mer.length(); i++) {
                    kmer_v[i] = base2code_dna.at(k_mer[i]);
                }
                at::Tensor kmer_t = torch::from_blob(kmer_v.data(), kmer_v.size(), torch::kInt64);
                if (sam_ptr->is_forward) kmer_t += 4;
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
                        if (dataQueue.size() > 1024) {
                            std::this_thread::sleep_for(std::chrono::seconds(10));
                        }
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
            if (dataQueue.size() > 1024) {
                std::this_thread::sleep_for(std::chrono::seconds(10));
            }
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
    total_cnt += thread_total_cnt;
    thread_cnt--;
}

void Data_Filter_get_features_for_model(Yao::Pod5Data p5,
                                        std::vector<Yao::SamRead *> inputs,
                                        std::queue<std::tuple<std::vector<std::string>, \
                                                std::vector<std::string>, \
                                                at::Tensor, at::Tensor>> &dataQueue,
                                        std::set<std::string> &sites,
                                        std::mutex &mtx1,
                                        std::condition_variable &cv1,
                                        int32_t batch_size,
                                        int32_t kmer_size,
                                        int32_t mapq_thresh_hold,
                                        float coverage_thresh_hold,
                                        float identity_thresh_hole,
                                        std::set<std::string> &motifset,
                                        size_t loc_in_motif,
                                        std::atomic<int64_t> &thread_cnt) {
    auto st = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> workers;
    size_t stride = (inputs.size() + 4) / 4;
    std::atomic<int32_t> total_cnt(0);

    for (size_t i = 0; i < inputs.size(); i += stride) {
        auto st = inputs.begin() + i;
        auto ed = inputs.begin() + std::min(i + stride, inputs.size());
        std::vector<Yao::SamRead*> input_s(st, ed);
        workers.emplace_back(Data_Filter_get_features_for_model_subthread,
                             std::ref(p5),
                             std::move(input_s),
                             std::ref(dataQueue),
                             std::ref(sites),
                             std::ref(mtx1),
                             std::ref(cv1),
                             batch_size,
                             kmer_size,
                             mapq_thresh_hold,
                             coverage_thresh_hold,
                             identity_thresh_hole,
                             std::ref(motifset),
                             loc_in_motif,
                             std::ref(total_cnt),
                             std::ref(thread_cnt));
    }
    // join all subthreads
    for (std::thread & worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }

    inputs.clear();
    p5.release();

    auto ed = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(ed - st);
    spdlog::info("Extract features for {} finished, extracted {} "
                 "features to module, cost {} seconds", p5.get_filename(),
                 total_cnt, (float)duration.count() / 1000.);
}

void Data_Filter_get_features_for_model_threadpool(int32_t num_workers,
                                                   fs::path &pod5_dir,
                                                   fs::path &bam_path,
                                                   Yao::Reference_Reader &ref,
                                                   std::queue<std::tuple<std::vector<std::string>,
                                                           std::vector<std::string>,
                                                           at::Tensor, at::Tensor>> &dataQueue,
                                                   std::set<std::string> &sites,
                                                   std::mutex &mtx1,
                                                   std::condition_variable &cv1,
                                                   int32_t batch_size,
                                                   int32_t kmer_size,
                                                   int32_t mapq_thresh_hold,
                                                   float coverage_thresh_hold,
                                                   float identity_thresh_hold,
                                                   std::set<std::string> &motifset,
                                                   size_t loc_in_motif) {
    spdlog::info("Start to get feature for Data Filter");
    const auto filename_to_path = Yao::get_filename_to_path(pod5_dir);
    std::atomic<int64_t> thread_cnt(0);
    {
        ThreadPool pool(num_workers * 2);
        const size_t buffer_size = 1024 * 1024 * 50;
        char *buffer = new char [buffer_size];
        std::string sam_str;
        std::string cmd = "samtools view -@ 12 " + bam_path.string();
        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);

        if (!pipe) {
            throw std::runtime_error("popen() failed!");
        }

        int32_t file_cnt = 0;
        std::string file_name_hold = "";
        std::vector<Yao::SamRead*> inputs;
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
                    pool.enqueue(Data_Filter_get_features_for_model,
                                 std::move(p5),
                                 inputs,
                                 std::ref(dataQueue),
                                 std::ref(sites),
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
                pool.enqueue(Data_Filter_get_features_for_model,
                             std::move(p5),
                             inputs,
                             std::ref(dataQueue),
                             std::ref(sites),
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

void Data_Filter_Model_Inference(torch::jit::Module &module,
                                 std::queue<std::tuple<std::vector<std::string>,
                                         std::vector<std::string>,
                                         at::Tensor, at::Tensor>> &dataQueue,
                                 std::queue<std::tuple<std::vector<at::Tensor>,
                                         std::vector<at::Tensor>,
                                         std::vector<float>>> &data_queue,
                                 std::mutex &mtx1,
                                 std::condition_variable &cv1,
                                 std::mutex &mtx2,
                                 std::condition_variable &cv2,
                                 size_t batch_size) {
    std::vector<torch::jit::IValue> inputs;
    using namespace torch::indexing;
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
        at::Tensor logits_v = logits.index({Slice{}, Slice{1,None}});
        logits_v = logits_v.contiguous();
        std::vector<float> p_rate_v(logits_v.data_ptr<float>(),
                                    logits_v.data_ptr<float>() + logits_v.numel());
        inputs.clear();

        kmer.to(torch::kCPU);
        signal.to(torch::kCPU);
        std::vector<at::Tensor> filtered_kmer;
        std::vector<at::Tensor> filtered_signal;
        std::vector<float> filtered_label;
        for (size_t i = 0; i < p_rate_v.size(); i ++) {
            if ((p_rate_v[i] <= 0.01) || (p_rate_v[i] >= 0.99)) {
                filtered_kmer.push_back(kmer[i]);
                filtered_signal.push_back(signal[i]);
                float label = (p_rate_v[i] <= 0.5) ? 0 : 1;
                filtered_label.push_back(label);
            }
        }
        {
            std::unique_lock<std::mutex> lock2(mtx2);
            if (data_queue.size() > 1024) {
                std::this_thread::sleep_for(std::chrono::seconds(10));
            }
            data_queue.push({filtered_kmer, filtered_signal, filtered_label});
        }
        cv2.notify_one();
        cnt += 1;
    }
    std::vector<float> filtered_label;
    filtered_label.push_back(-1);
    std::unique_lock<std::mutex> lock2(mtx2);
    data_queue.push({std::vector<at::Tensor>(), std::vector<at::Tensor>(), filtered_label});
    lock2.unlock();
    cv2.notify_one();
//    auto thread_id = std::this_thread::get_id(); // convert thread id to string
//    std::stringstream sstream;
//    sstream << thread_id;
//    std::string thread_str = sstream.str();
    spdlog::info("Model inference thread finished, processed {} batch", cnt);
}

void Data_Filter_write_data(fs::path &write_dir,
                            std::queue<std::tuple<std::vector<at::Tensor>,
                                    std::vector<at::Tensor>,
                                    std::vector<float>>> &data_queue,
                            std::mutex &mtx2,
                            std::condition_variable &cv2) {
    int64_t chunk_id = -1;
    fs::path write_file = write_dir / ("chunk_" + std::to_string(chunk_id));
    uint64_t F_size = (uint64_t )15 * 1024 * 1024 * 1024;
    size_t batch_size = (size_t)5 * 1024 * 1024 * 1021;
//    uint64_t F_size = (uint64_t )10 * 1024 * 1024;
    std::vector<float> data;
    while (true) {
        std::unique_lock<std::mutex> lock(mtx2);
        cv2.wait(lock, [&data_queue] {return !data_queue.empty();});
        auto [kmer_vs, signal_vs, labels] = data_queue.front();
        data_queue.pop();
        lock.unlock();
        if (labels.size() == 0 || (labels[0] < 0)) break;
        for (size_t i = 0; i < labels.size(); i++) {
            at::Tensor kmer = kmer_vs[i];

            kmer = kmer.to(torch::kFloat32);
            std::vector<float> kmer_v(kmer.data_ptr<float>(), kmer.data_ptr<float>() + kmer.numel());
            std::vector<float> signal_v(signal_vs[i].data_ptr<float>(), signal_vs[i].data_ptr<float>() + signal_vs[i].numel());
            data.insert(data.end(), kmer_v.begin(), kmer_v.end());
            data.insert(data.end(), signal_v.begin(), signal_v.end());
            data.push_back(labels[i]);
        }
        size_t len = kmer_vs[0].size(0) * 20 + 1;

        if (data.size() > batch_size){
            if (fs::exists(write_file) && fs::file_size(write_file) < F_size) {
                // write npy of mode "a"
                cnpy::npy_save(write_file.string(), &data[0], {data.size() / len, len}, "a");
            } else {
                // create a new chunk
                chunk_id++;
                write_file = write_dir / ("chunk_" + std::to_string(chunk_id));
                // write npy of mode "w"
                cnpy::npy_save(write_file.string(), &data[0], {data.size() / len, len}, "w");
            }
            data.clear();
        }
    }
}
