//
// Created by dell on 2023/8/14.
//

#ifndef BAMCLASS_UTILS_THREAD_H
#define BAMCLASS_UTILS_THREAD_H

#pragma once
#include <mutex>
#include <map>
#include <set>
#include <condition_variable>
#include "../DataLoader/Pod5Data.h"
#include "../DataLoader/SamRead.h"
#include "../utils/utils_func.h"


namespace Yao{

// sub thread from thread pool to extract hc features
// the idea to use split a pod5 read to four sub-thread is to
// make use of multi CPU and reduce memory cost
void get_hc_features_subthread(Yao::Pod5Data &p5,
                               std::vector<Yao::SamRead*> inputs,
                               fs::path & writefile,
                               std::set<std::string> & pos_hc_sites,
                               std::set<std::string> & neg_hc_sites,
                               std::mutex &mtx,
                               std::condition_variable &cv,
                               int32_t kmer_size,
                               int32_t mapq_thresh_hold,
                               float coverage_thresh_hold,
                               float identity_thresh_hold,
                               std::set<std::string> & motifset,
                               size_t loc_in_motif,
                               std::atomic<int64_t> &total_cnt, 
                               std::atomic<int64_t> &thread_cnt);

void get_hc_features(Yao::Pod5Data p5,
                       std::vector<Yao::SamRead*> inputs,
                       fs::path writefile,
                       std::set<std::string> & pos_hc_sites,
                       std::set<std::string> & neg_hc_sites,
                       int32_t kmer_size,
                       int32_t mapq_thresh_hold,
                       float coverage_thresh_hold,
                       float identity_thresh_hole,
                       std::set<std::string> & motifset,
                       size_t loc_in_motif, 
                       std::atomic<int64_t> &thread_cnt);

void get_feature_for_model_subthread(Yao::Pod5Data &p5,
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
                                     std::set<std::string>& motifset,
                                     size_t loc_in_motif,
                                     std::atomic<int32_t> &total_cnt,
                                     std::atomic<int64_t> &thread_cnt);


void get_feature_for_model(Yao::Pod5Data p5,
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
                           std::atomic<int64_t>& thread_cnt) ;

// function to extract features and pass it to model
// uses a thread pool to accelerate feature extraction
void get_feature_for_model_with_thread_pool(int32_t num_workers,
                                            fs::path & pod5_dir,
                                            fs::path & bam_path,
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
                                            size_t loc_in_motif);

// model running thread, get data from function `get_feature_for_model_with_thread_pool`
void Model_Inference(torch::jit::script::Module &module,
                     std::queue<std::tuple<std::vector<std::string>,\
                                                       std::vector<std::string>, \
                                                       at::Tensor, at::Tensor>> &dataQueue,
                     std::queue<std::vector<std::string>> & site_key_Queue,
                     std::queue<std::vector<std::string>> &site_info_Queue,
                     std::queue<std::vector<int64_t>> &pred_Queue,
                     std::queue<std::vector<float>> & p_rate_Queue,
                     std::mutex &mtx1,
                     std::condition_variable &cv1,
                     std::mutex &mtx2,
                     std::condition_variable &cv2,
                     size_t batch_size);

// process result from model
// the result will write to two files
// one for accumulate methylation frequency
// the other is detailed methylation rate on every single read
void count_modification_thread(std::map<std::string, std::vector<float>> &site_dict,
                               std::queue<std::vector<std::string>> &site_key_Queue,
                               std::queue<std::vector<std::string>> &site_info_Queue,
                               std::queue<std::vector<int64_t>> &pred_Queue,
                               std::queue<std::vector<float>> & p_rate_Queue,
                               fs::path &write_file2,
                               std::mutex &mtx2,
                               std::condition_variable &cv2);


}

#endif //BAMCLASS_UTILS_THREAD_H
