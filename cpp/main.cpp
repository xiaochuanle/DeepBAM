#include <iostream>
#include <filesystem>
#include <chrono>
#include <spdlog/spdlog.h>

#include "DataLoader/Reference_Reader.h"
#include "utils/utils_func.h"
#include "3rdparty/argparse/argparse.h"
#include "Feature_Extractor/Feature_Extractor.h"
#include "Call_Modification/Call_Modification.h"

namespace fs = std::filesystem;

int main(int argc, char **argv) {
    argparse::ArgumentParser program("BamClass", "0.0.1");

    argparse::ArgumentParser extract_hc_sites("extract_hc_sites");
    extract_hc_sites.add_description("extract features for model training with "\
                                     "high confident bisulfite data");
    extract_hc_sites.add_argument("pod5_dir")
            .help("path to pod5 directory");
    extract_hc_sites.add_argument("bam_path")
            .help("path to bam file, sorted by file name is needed");
    extract_hc_sites.add_argument("reference_path")
            .help("path to reference genome");
    extract_hc_sites.add_argument("ref_type")
            .default_value("DNA")
            .help("reference genome tyoe");
    extract_hc_sites.add_argument("write_dir")
            .help("write directory, write file format ${pod5filename}.npy");
    extract_hc_sites.add_argument("pos")
            .help("positive high accuracy methylation sites");
    extract_hc_sites.add_argument("neg")
            .help("negative high accuracy methylation sites");
    extract_hc_sites.add_argument("kmer_size")
            .help("kmer size for extract features")
            .default_value((int32_t) 51)
            .scan<'i', int>();
    extract_hc_sites.add_argument("num_workers")
            .scan<'i', int>()
            .default_value(10)
            .help("num of workers in extract feature thread pool, every worker"
                  "possess one pod5 file and its corresponding sam reads");
    extract_hc_sites.add_argument("sub_thread_per_worker")
        .scan<'i', int>()
        .default_value(4)
        .help("num of sub thread per worker");
    extract_hc_sites.add_argument("motif_type")
            .default_value("CG")
            .help("motif_type default CG");
    extract_hc_sites.add_argument("loc_in_motif")
            .scan<'i', int>()
            .help("Location in motifset");


    argparse::ArgumentParser extract_and_call_mods("extract_and_call_mods");
    extract_and_call_mods.add_description("asynchronously extract features and"
                                          " pass data to model to get modification result");
    extract_and_call_mods.add_argument("pod5_dir")
            .help("path to pod5 directory");
    extract_and_call_mods.add_argument("bam_path")
            .help("path to bam file, sorted by file name is needed");
    extract_and_call_mods.add_argument("reference_path")
            .help("path to reference genome");
    extract_and_call_mods.add_argument("ref_type")
            .default_value("DNA")
            .help("reference genome type");
//    extract_and_call_mods.add_argument("write_file1")
//            .help("write modification count file path");
    extract_and_call_mods.add_argument("write_file")
            .help("write detailed modification result file path");
    extract_and_call_mods.add_argument("module_path")
            .help("module path to trained model");
    extract_and_call_mods.add_argument("kmer_size")
            .help("kmer size for extract features")
            .default_value((int32_t) 51)
            .scan<'i', int>();
    extract_and_call_mods.add_argument("num_workers")
            .scan<'i', int>()
            .default_value(10)
            .help("num of workers in extract feature thread pool, every worker"
                  "possess one pod5 file and its corresponding sam reads");
    extract_and_call_mods.add_argument("sub_thread_per_worker")
            .scan<'i', int>()
            .default_value(4)
            .help("num of sub thread per worker");
    extract_and_call_mods.add_argument("batch_size")
            .scan<'i', int>()
            .default_value(4096)
            .help("default batch size");
    extract_and_call_mods.add_argument("motif_type")
            .default_value("CG")
            .help("motif_type default CG");
    extract_and_call_mods.add_argument("loc_in_motif")
            .scan<'i', int>()
            .help("Location in motifset");




    program.add_subparser(extract_hc_sites);
    program.add_subparser(extract_and_call_mods);


    try {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    if (program.is_subcommand_used("extract_hc_sites")) {
        spdlog::info("Bam Class mode: extract hc sites");
        fs::path pod5_dir = extract_hc_sites.get<std::string>("pod5_dir");
        fs::path bam_path = extract_hc_sites.get<std::string>("bam_path");
        fs::path reference_path = extract_hc_sites.get<std::string>("reference_path");
        std::string ref_type = extract_hc_sites.get<std::string>("ref_type");
        fs::path write_dir = extract_hc_sites.get<std::string>("write_dir");
        fs::path pos = extract_hc_sites.get<std::string>("pos");
        fs::path neg = extract_hc_sites.get<std::string>("neg");
        int32_t kmer_size = extract_hc_sites.get<int32_t>("kmer_size");
        size_t num_workers = extract_hc_sites.get<int32_t>("num_workers");
        size_t num_sub_th = extract_hc_sites.get<int32_t>("sub_thread_per_worker");
        std::string motiftype = extract_hc_sites.get<std::string>("motif_type");
        size_t loc_in_motif = extract_hc_sites.get<int32_t>("loc_in_motif");

        auto pos_hc_sites = Yao::get_hc_set(pos);
        auto neg_hc_sites = Yao::get_hc_set(neg);
        auto motifset = Yao::get_motif_set(motiftype);

        Yao::Feature_Extractor feature_extractor(pod5_dir,
                                                 reference_path,
                                                 ref_type);

        feature_extractor.extract_hc_sites(
                num_workers,
                num_sub_th,
                bam_path,
                write_dir,
                pos_hc_sites,
                neg_hc_sites,
                motifset,
                loc_in_motif,
                kmer_size
        );
    } else if (program.is_subcommand_used("extract_and_call_mods")) {
        spdlog::info("Bam Class mode: extract and call mods");
        fs::path pod5_dir = extract_and_call_mods.get<std::string>("pod5_dir");
        fs::path bam_path = extract_and_call_mods.get<std::string>("bam_path");
        fs::path reference_path = extract_and_call_mods.get<std::string>("reference_path");
        std::string ref_type = extract_and_call_mods.get<std::string>("ref_type");
//        fs::path write_file1 = extract_and_call_mods.get<std::string>("write_file1");
        fs::path write_file = extract_and_call_mods.get<std::string>("write_file");
        fs::path module_path = extract_and_call_mods.get<std::string>("module_path");
        int32_t kmer_size = extract_and_call_mods.get<int32_t>("kmer_size");
        size_t num_workers = extract_and_call_mods.get<int32_t>("num_workers");
        size_t sub_thread = extract_and_call_mods.get<int32_t>("sub_thread_per_worker");
        int32_t batch_size = extract_and_call_mods.get<int32_t>("batch_size");
        std::string motif_type = extract_and_call_mods.get<std::string>("motif_type");
        size_t loc_in_motif = extract_and_call_mods.get<int32_t>("loc_in_motif");

        auto st = std::chrono::high_resolution_clock::now();

        auto motifset = Yao::get_motif_set(motif_type);

        Yao::Call_Modification caller(batch_size,
                                      kmer_size,
                                      reference_path,
                                      ref_type,
                                      module_path);

        caller.call_mods(pod5_dir,
                         bam_path,
                         write_file,
                         num_workers,
                         sub_thread,
                         motifset,
                         loc_in_motif
        );

        spdlog::info("Extract feature and call mods finished");
        auto ed = std::chrono::high_resolution_clock::now();
        auto d = std::chrono::duration_cast<std::chrono::seconds>(ed - st);
        spdlog::info("Total time cost {} seconds", d.count());
    } else {
        spdlog::info("This is a tool for extract features for model training, "\
        "or call modification with trained model, type `-h` for further guide");
    }
    return 0;
}