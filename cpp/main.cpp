#include <iostream>
#include <filesystem>
#include <chrono>
#include <spdlog/spdlog.h>

#include "DataLoader/Reference_Reader.h"
#include "utils/utils_func.h"
#include "3rdparty/argparse/argparse.h"
#include "Feature_Extractor/Feature_Extractor.h"
#include "Call_Modification/Call_Modification.h"
#include "Data_Filter/Data_Filter.h"

namespace fs = std::filesystem;

int main(int argc, char **argv) {
    argparse::ArgumentParser program("DeepBam","1");

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
            .default_value((int32_t)51)
            .scan<'i', int>();
    extract_hc_sites.add_argument("num_workers")
            .scan<'i', int>()
            .default_value(10)
            .help("num of workers in extract feature thread pool, "\
    "every thread contains 4 sub-threads to extract "\
    "features, so do not choose more than (num of cpu / 4) threads");

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
    extract_and_call_mods.add_argument("write_file1")
            .help("write modification count file path");
    extract_and_call_mods.add_argument("write_file2")
            .help("write detailed modification result file path");
    extract_and_call_mods.add_argument("module_path")
            .help("module path to trained model");
    extract_and_call_mods.add_argument("kmer_size")
            .help("kmer size for extract features")
            .default_value((int32_t)51)
            .scan<'i', int>();
    extract_and_call_mods.add_argument("num_workers")
            .scan<'i', int>()
            .default_value(10)
            .help("num of workers in extract feature thread pool, "\
    "every thread contains 4 sub-threads to extract "\
    "features, so do not choose more than (num of cpu / 4) threads");
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

    argparse::ArgumentParser filter_data("filter_data");
    filter_data.add_description("This is a test idea designed to use a model trained "
                                "on (0.99-0.02) hc sites, and use it to filter high accuracy sites from "
                                "(0.2-0.8), and retrain the model on high accuracy data between (0.2-0.8) "
                                "and (0.99-0.02)");
    filter_data.add_argument("pod5_dir")
            .help("path to pod5 directory");
    filter_data.add_argument("bam_path")
            .help("path to bam file, sorted by file name is needed");
    filter_data.add_argument("reference_path")
            .help("path to reference genome");
    filter_data.add_argument("ref_type")
            .default_value("DNA")
            .help("reference genome type");
    filter_data.add_argument("write_dir")
            .help("write directory, write file format ${pod5filename}.npy");
    filter_data.add_argument("sites")
            .help("sites that contain methylation frequency between 0.2 and 0.8");
    filter_data.add_argument("module_path")
            .help("module path to trained model");
    filter_data.add_argument("kmer_size")
            .help("kmer size for extract features")
            .default_value((int32_t)51)
            .scan<'i', int>();
    filter_data.add_argument("num_workers")
            .scan<'i', int>()
            .default_value(10)
            .help("num of workers in extract feature thread pool, "
                  "every thread contains 4 sub-threads to extract "
                  "features, so do not choose more than (num of cpu / 4) threads");
    filter_data.add_argument("batch_size")
            .scan<'i', int>()
            .default_value(4096)
            .help("default batch size");
    filter_data.add_argument("motif_type")
            .default_value("CG")
            .help("motif_type default CG");
    filter_data.add_argument("loc_in_motif")
            .scan<'i', int>()
            .help("Location in motifset");



    program.add_subparser(extract_hc_sites);
    program.add_subparser(extract_and_call_mods);
    program.add_subparser(filter_data);


    try {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    if (program.is_subcommand_used("extract_hc_sites")) {
        spdlog::info("DeepBam mode: extract hc sites");
        fs::path pod5_dir = extract_hc_sites.get<std::string>("pod5_dir");
        fs::path bam_path = extract_hc_sites.get<std::string>("bam_path");
        fs::path reference_path = extract_hc_sites.get<std::string>("reference_path");
        std::string ref_type = extract_hc_sites.get<std::string>("ref_type");
        fs::path write_dir = extract_hc_sites.get<std::string>("write_dir");
        fs::path pos = extract_hc_sites.get<std::string>("pos");
        fs::path neg = extract_hc_sites.get<std::string>("neg");
        int32_t kmer_size = extract_hc_sites.get<int32_t>("kmer_size");
        size_t num_workers = extract_hc_sites.get<int32_t>("num_workers");
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
                bam_path,
                write_dir,
                pos_hc_sites,
                neg_hc_sites,
                motifset,
                loc_in_motif,
                kmer_size
                );
    }
    else if (program.is_subcommand_used("extract_and_call_mods")) {
        spdlog::info("DeepBam mode: extract and call mods");
        fs::path pod5_dir = extract_and_call_mods.get<std::string>("pod5_dir");
        fs::path bam_path = extract_and_call_mods.get<std::string>("bam_path");
        fs::path reference_path = extract_and_call_mods.get<std::string>("reference_path");
        std::string ref_type = extract_and_call_mods.get<std::string>("ref_type");
        fs::path write_file1 = extract_and_call_mods.get<std::string>("write_file1");
        fs::path write_file2 = extract_and_call_mods.get<std::string>("write_file2");
        fs::path module_path = extract_and_call_mods.get<std::string>("module_path");
        int32_t kmer_size = extract_and_call_mods.get<int32_t>("kmer_size");
        size_t num_workers = extract_and_call_mods.get<int32_t>("num_workers");
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
                         write_file1,
                         write_file2,
                         num_workers,
                         motifset,
                         loc_in_motif
                         );

        spdlog::info("Extract feature and call mods finished");
        auto ed = std::chrono::high_resolution_clock::now();
        auto d = std::chrono::duration_cast<std::chrono::seconds>(ed - st);
        spdlog::info("Total time cost {} seconds", d.count());
    }
    else if (program.is_subcommand_used("filter_data")) {
        spdlog::info("DeepBam mode: filter data");
        fs::path pod5_dir = filter_data.get<std::string>("pod5_dir");
        fs::path bam_path = filter_data.get<std::string>("bam_path");
        fs::path reference_path = filter_data.get<std::string>("reference_path");
        std::string ref_type = filter_data.get<std::string>("ref_type");
        fs::path write_dir = filter_data.get<std::string>("write_dir");
        fs::path site_path = filter_data.get<std::string>("sites");
        fs::path module_path = filter_data.get<std::string>("module_path");
        int32_t kmer_size = filter_data.get<int32_t>("kmer_size");
        size_t num_workers = filter_data.get<int32_t>("num_workers");
        int32_t batch_size = filter_data.get<int32_t>("batch_size");
        std::string motif_type = filter_data.get<std::string>("motif_type");
        size_t loc_in_motif = filter_data.get<int32_t>("loc_in_motif");

        std::set<std::string> motifset = Yao::get_motif_set(motif_type);
        std::set<std::string> sites = Yao::get_hc_set(site_path);

        Yao::Data_Filter data_filter (batch_size,
                                      kmer_size,
                                      reference_path,
                                      ref_type,
                                      module_path);

        data_filter.filter_data(pod5_dir,
                                bam_path,
                                write_dir,
                                sites,
                                num_workers,
                                motifset,
                                loc_in_motif);
    }
    else {
        spdlog::info("This is a tool for extract features for model training, "\
        "or call modification with trained model, type `-h` for further guide");
    }
    return 0;
}