//
// Created by dell on 2023/8/17.
//
#include <torch/script.h>
#include <spdlog/spdlog.h>
#include "Call_Modification.h"
#include "../utils/utils_thread.h"
#include "../utils/utils_func.h"

Yao::Call_Modification::Call_Modification(size_t batch_size_,
                                     size_t kmer_size_,
                                     fs::path reference_path_,
                                     std::string ref_type,
                                     fs::path module_path):
                                     reference_path(reference_path_),
                                     ref(std::move(Yao::Reference_Reader(reference_path, ref_type))),
                                     batch_size(batch_size_),
                                     kmer_size(kmer_size_)
                                     {
    try {
        module = torch::jit::load(module_path);
        spdlog::info("Successfully load module!");
    }
    catch (const c10::Error& e) {
        spdlog::error("Error loading the module");
    }
    module.to(torch::kCUDA);

    try{
        at::Tensor kmer = torch::randint(0,8,{512, 51}, torch::kLong);
        at::Tensor signal = torch::rand({512,1,51, 19}, torch::kFloat32);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(kmer.to(torch::kCUDA));
        inputs.push_back(signal.to(torch::kCUDA));
        auto result = module.forward(inputs);
//        std::cout << "\n";
        spdlog::info("module test successfully!");
    }
    catch (const c10::Error &e ) {
        spdlog::error("error loading the module");
        std::cout << e.what() << std::endl;
    }

    // default parameter to filter reads
    mapq_thresh_hold = 20;
    coverage_thresh_hold = 0.8;
    identity_thresh_hold = 0.8;
}

void Yao::Call_Modification::call_mods(fs::path & pod5_dir,
                                       fs::path & bam_path,
                                       fs::path & write_file1,
                                       fs::path & write_file2,
                                       size_t & num_workers,
                                       std::set<std::string> & motifset,
                                       size_t & loc_in_motif) {
    auto start = std::chrono::high_resolution_clock::now();
    std::thread thread1(Yao::get_feature_for_model_with_thread_pool,
                        num_workers,
                        std::ref(pod5_dir),
                        std::ref(bam_path),
                        std::ref(ref),
                        std::ref(dataQueue),
                        std::ref(mtx1),
                        std::ref(cv1),
                        batch_size,
                        kmer_size,
                        mapq_thresh_hold,
                        coverage_thresh_hold,
                        identity_thresh_hold,
                        std::ref(motifset),
                        loc_in_motif
                        );
    std::thread thread2(Yao::Model_Inference,
                        std::ref(module),
                        std::ref(dataQueue),
                        std::ref(site_key_Queue),
                        std::ref(site_info_Queue),
                        std::ref(pred_Queue),
                        std::ref(p_rate_Queue),
                        std::ref(mtx1),
                        std::ref(cv1),
                        std::ref(mtx2),
                        std::ref(cv2),
                        batch_size);
    std::thread thread3(Yao::count_modification_thread,
                        std::ref(site_dict),
                        std::ref(site_key_Queue),
                        std::ref(site_info_Queue),
                        std::ref(pred_Queue),
                        std::ref(p_rate_Queue),
                        std::ref(write_file2),
                        std::ref(mtx2),
                        std::ref(cv2));
    thread1.join();
    thread2.join();
    thread3.join();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

    spdlog::info("Total time taken by extract feature and call modification: {} seconds", duration.count());

    spdlog::info("Start to write modification result to file: {}", write_file1.string());
    std::ofstream out;
    out.open(write_file1);
    out << "chromosome\t" << "pos\t" << "strand\t" << "mod\t" << "count\n";
    for (const auto & [key, value] : site_dict) {
        float b = value[1], c = value[2];
        out << key << "\t"  << b / c << "\t" << c<< "\t" << std::endl;
    }
    out.close();
    spdlog::info("Write result finished");
}
