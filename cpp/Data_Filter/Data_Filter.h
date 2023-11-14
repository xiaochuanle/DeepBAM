//
// Created by dell on 2023/9/21.
//

#ifndef BAMCLASS_DATA_FILTER_H
#define BAMCLASS_DATA_FILTER_H
#include <torch/torch.h>
#include <mutex>
#include <map>
#include <set>
#include <condition_variable>
#include "../DataLoader/Pod5Data.h"
#include "../DataLoader/SamRead.h"
#include "../utils/utils_func.h"
#include "../DataLoader/Reference_Reader.h"

void Data_Filter_get_features_for_model_subthread(Yao::Pod5Data &p5,
                                                  std::vector<Yao::SamRead *> inputs,
                                                  std::queue<std::tuple<std::vector<std::string>,\
                                                         std::vector<std::string>, \
                                                         at::Tensor, at::Tensor>> &dataQueue,
                                                  std::set<std::string> &sites,
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
void Data_Filter_get_features_for_model(Yao::Pod5Data p5,
                                        std::vector<Yao::SamRead *> inputs,
                                        std::queue<std::tuple<std::vector<std::string>,\
                                                 std::vector<std::string>, \
                                                 at::Tensor, at::Tensor>> &dataQueue,
                                        std::set<std::string> &sites,
                                        std::mutex &mtx1,
                                        std::condition_variable &cv1,
                                        int32_t batch_size,
                                        int32_t kmer_size,
                                        int32_t mapq_thresh_hold,
                                        float coverage_thresh_hold,
                                        float identity_thresh_hold,
                                        std::set<std::string> & motifset,
                                        size_t loc_in_motif,
                                        std::atomic<int64_t>& thread_cnt);

void Data_Filter_get_features_for_model_threadpool(int32_t num_workers,
                                                   fs::path &pod5_dir,
                                                   fs::path &bam_path,
                                                   Yao::Reference_Reader & ref,
                                                   std::queue<std::tuple<std::vector<std::string>,\
                                                         std::vector<std::string>, \
                                                         at::Tensor, at::Tensor>> &dataQueue,
                                                   std::set<std::string> &sites,
                                                   std::mutex &mtx1,
                                                   std::condition_variable &cv1,
                                                   int32_t batch_size,
                                                   int32_t kmer_size,
                                                   int32_t mapq_thresh_hold,
                                                   float coverage_thresh_hold,
                                                   float identity_thresh_hold,
                                                   std::set<std::string> & motifset,
                                                   size_t loc_in_motif);

void Data_Filter_Model_Inference(torch::jit::script::Module &module,
                     std::queue<std::tuple<std::vector<std::string>,\
                                                       std::vector<std::string>, \
                                                       at::Tensor, at::Tensor>> &dataQueue,
                     std::queue<std::tuple<std::vector<at::Tensor>,
                             std::vector<at::Tensor>,
                             std::vector<float>>> &data_queue,
//                     std::queue<std::vector<float>> & p_rate_Queue,
                     std::mutex &mtx1,
                     std::condition_variable &cv1,
                     std::mutex &mtx2,
                     std::condition_variable &cv2,
                     size_t batch_size);

void Data_Filter_write_data(fs::path &write_dir,
                            std::queue<std::tuple<std::vector<at::Tensor>,
                                    std::vector<at::Tensor>,
                                    std::vector<float>>> &data_queue,
                            std::mutex &mtx2,
                            std::condition_variable &cv2);

namespace  Yao {
class Data_Filter {
public:
    Data_Filter(size_t batch_size_,
                size_t kmer_size_,
                fs::path reference_path_,
                std::string ref_type,
                fs::path module_path);

    void filter_data(fs::path & pod5_dir,
                     fs::path & bam_path,
                     fs::path & write_dir,
                     std::set<std::string> & sites,
                     size_t & num_workers,
                     std::set<std::string> & motifset,
                     size_t & loc_in_motif);

    ~Data_Filter()=default;

private:
    std::queue<std::tuple<std::vector<std::string>, std::vector<std::string> ,at::Tensor, at::Tensor>> dataQueue;
    std::queue<std::vector<std::string>> site_key_Queue;
    std::queue<std::vector<std::string>> site_info_Queue;
    std::queue<std::vector<int64_t>> pred_Queue;
    std::queue<std::vector<float>> p_rate_Queue;
    std::queue<std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>, std::vector<float>>> data_queue;
    std::mutex mtx1;
    std::mutex mtx2;
    std::condition_variable cv1;
    std::condition_variable cv2;
    torch::jit::script::Module module;
    fs::path reference_path;
    Yao::Reference_Reader ref;
    std::string file_name_hold;
    size_t batch_size;
    std::map<std::string, std::vector<float>> site_dict;
    size_t kmer_size;
    int32_t mapq_thresh_hold;
    float coverage_thresh_hold;
    float identity_thresh_hold;
};

}


#endif //BAMCLASS_DATA_FILTER_H
