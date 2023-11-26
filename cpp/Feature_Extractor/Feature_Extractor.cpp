//
// Created by dell on 2023/8/11.
//

#include <chrono>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <spdlog/spdlog.h>
#include "Feature_Extractor.h"
#include "../utils/utils_func.h"
#include "../3rdparty/threadpool/threadpool.h"
#include "../utils/utils_thread.h"
#include "../DataLoader/Reference_Reader.h"

Yao::Feature_Extractor::Feature_Extractor(fs::path pod5_dir_,
          fs::path reference_path_,
          std::string ref_type):
        pod5_dir(pod5_dir_),
        reference_path(reference_path_),
        ref(Yao::Reference_Reader(reference_path, ref_type)){
    filename_to_path = Yao::get_filename_to_path(pod5_dir);
    file_name_hold = "";

    mapq_thresh_hold = 20;
    coverage_thresh_hold = 0.8;
    identity_thresh_hold = 0.8;
}


void Yao::Feature_Extractor::extract_hc_sites(size_t num_workers,
                                              fs::path &bam_file,
                                              fs::path &write_dir,
                                              std::set<std::string> &pos_hc_sites,
                                              std::set<std::string> &neg_hc_sites,
                                              std::set<std::string> & motifset,
                                              int32_t loc_in_motif,
                                              int32_t kmer_size) {
    auto st = std::chrono::high_resolution_clock::now();
    auto reformat_chr = ref.reformat_chr();
    std::atomic<int64_t> thread_cnt(0);
    {

        ThreadPool pool(num_workers * 2);
        
        const size_t buffer_size = 1024 * 1024 * 50;
        char * buffer = new char [buffer_size];

        std::string sam_str;
        std::string cmd = "samtools view -@ 12 " + bam_file.string();
        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);

        if (!pipe) {
            throw std::runtime_error("popen() failed!");
        }

        int32_t file_cnt = 0;
        std::vector<Yao::SamRead *> inputs;
        while (fgets(buffer, buffer_size, pipe.get()) != nullptr) {
            sam_str = buffer;
            Yao::SamRead *sam_ptr = new Yao::SamRead(sam_str);
            if (file_name_hold != sam_ptr->file_name && !file_name_hold.empty()) {

                fs::path p5_file;
                try{
                    p5_file = filename_to_path.at(file_name_hold);
                    fs::path write_file = write_dir / p5_file.filename();
                    write_file.replace_extension(".npz");
                    Yao::Pod5Data p5(p5_file);
                    while (thread_cnt >= (int64_t)(num_workers * 3)) {
                        std::this_thread::sleep_for(std::chrono::seconds(1));
                    }

                    pool.enqueue(Yao::get_hc_features,
                                 p5,
                                 inputs,
                                 std::ref(reformat_chr),
                                 write_file,
                                 std::ref(pos_hc_sites),
                                 std::ref(neg_hc_sites),
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
                         " thread pool, progress [{}/{}]",
                         p5.get_filename(), file_cnt, filename_to_path.size());
                }
                catch (...) {
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
        if (inputs.size() > 0) {
            fs::path p5_file;
            try{
                p5_file = filename_to_path.at(file_name_hold);
                fs::path write_file = write_dir / p5_file.filename();
                write_file.replace_extension(".npz");
                Yao::Pod5Data p5(p5_file);
                pool.enqueue(Yao::get_hc_features,
                             p5,
                             inputs,
                             std::ref(reformat_chr),
                             write_file,
                             std::ref(pos_hc_sites),
                             std::ref(neg_hc_sites),
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
                         " thread pool, progress [{}/{}]",
                             p5.get_filename(), file_cnt, filename_to_path.size());
            }
            catch (...) {
                spdlog::error("Couldn't find file: {}", file_name_hold);
                // release allocated memory
                for (auto &ptr : inputs) {
                    delete ptr;
                }
                inputs.clear();
            }
        }
        delete [] buffer;
    }
    auto ed = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(ed - st);
    spdlog::info("extract hc features finished, cost {} seconds", duration.count());
}
