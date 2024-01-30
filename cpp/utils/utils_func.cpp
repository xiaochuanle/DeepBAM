//
// Created by dell on 2023/8/10.
//
#include <torch/torch.h>
#include <algorithm>
#include <random>
#include <set>
#include <spdlog/spdlog.h>
#include "utils_func.h"

/*
 * generate motifs by giving motif_type
 * current version {"CG", "CHG", "CHH"}
 * */
std::set<std::string> Yao::get_motif_set(std::string &motif_type) {
    if (motif_type == "CG") {
        return {"CG"};
    } else if (motif_type == "CHG") {
        return {"CAG", "CTG", "CCG"};
    } else if (motif_type == "CHH") {
        // depth first search ......
        std::set<std::string> motifset;
        std::string extend = "ACT";
        for (size_t i = 0; i < extend.length(); i++) {
            std::string base = "C";
            base.push_back(extend[i]);
            for (size_t j = 0; j < extend.length(); j++) {
                base.push_back(extend[j]);
                motifset.insert(base);
                base.pop_back();
            }
            base.pop_back();
        }
        return motifset;
    }
    return {};
}

/*
* mapping kmer-seq to number
* */
std::map<char, int> Yao::get_base2code_dna() {

    std::map<char, int> map;
    map['A'] = 0;
    map['C'] = 1;
    map['G'] = 2;
    map['T'] = 3;
    map['N'] = 4;
    map['W'] = 5;
    map['S'] = 6;
    map['M'] = 7;
    map['K'] = 8;
    map['R'] = 9;
    map['Y'] = 10;
    map['B'] = 11;
    map['V'] = 12;
    map['D'] = 13;
    map['H'] = 14;
    map['Z'] = 15;
    return map;
}

/*
 * mapping kmer-seq to number
 * */
std::map<char, int> Yao::get_base2code_rna() {
    std::map<char, int> map;
    map['A'] = 0;
    map['C'] = 1;
    map['G'] = 2;
    map['U'] = 3;
    map['N'] = 4;
    map['W'] = 5;
    map['S'] = 6;
    map['M'] = 7;
    map['K'] = 8;
    map['R'] = 9;
    map['Y'] = 10;
    map['B'] = 11;
    map['V'] = 12;
    map['D'] = 13;
    map['H'] = 14;
    map['Z'] = 15;
    return map;
}

/*
 * map dna bases to their complement bases
 * */
std::map<char, char> Yao::get_basepairs_dna() {
    std::map<char, char> map;
    map['A'] = 'T';
    map['C'] = 'G';
    map['G'] = 'C';
    map['T'] = 'A';
    map['N'] = 'N';
    map['W'] = 'W';
    map['S'] = 'S';
    map['M'] = 'K';
    map['K'] = 'M';
    map['R'] = 'Y';
    map['Y'] = 'R';
    map['B'] = 'V';
    map['V'] = 'B';
    map['D'] = 'H';
    map['H'] = 'D';
    map['Z'] = 'Z';
    return map;
}

/*
 * map rna bases to their complement bases
 * */
std::map<char, char> Yao::get_basepairs_rna() {
    std::map<char, char> map;
    map['A'] = 'U';
    map['C'] = 'G';
    map['G'] = 'C';
    map['U'] = 'A';
    map['N'] = 'N';
    map['W'] = 'W';
    map['S'] = 'S';
    map['M'] = 'K';
    map['K'] = 'M';
    map['R'] = 'Y';
    map['Y'] = 'R';
    map['B'] = 'V';
    map['V'] = 'B';
    map['D'] = 'H';
    map['H'] = 'D';
    map['Z'] = 'Z';
    return map;
}

void Yao::get_movetable(std::vector<uint64_t> &movetable, std::string &movetable_str) {
    // movetable info, record 1's location
    assert(movetable_str[9] == '1');
    for (size_t i = 9; i < movetable_str.length(); i += 2) {
        if (movetable_str[i] == '1') {
            movetable.push_back((i - 9) / 2);
        }
    }
    // `(movetable_str.length() - 8) / 2`, represents total length of movetable
    movetable.push_back((movetable_str.length() - 8) / 2);
}

void Yao::get_refloc_of_methysite_in_motif(std::vector<int32_t> &refloc_of_methysite,
                                           std::string &seqstr,
                                           std::set<std::string> &motifset,
                                           size_t loc_in_motif) {
    auto len = (*motifset.begin()).length();
    for (size_t i = 0; i < seqstr.length(); i++) {
        if (motifset.find(seqstr.substr(i, len)) != motifset.end()) {
            refloc_of_methysite.push_back(i + loc_in_motif);
        }
    }
}

void Yao::group_signal_by_movetable(std::vector<at::Tensor> &signal_group,
                                    at::Tensor &trimed_signal,
                                    std::vector<uint64_t> &movetable,
                                    size_t stride) {
    // TODO: signal_duration not exactly equal to call_events * stride
    // TODO: maybe this move * stride way is not right!
    assert(movetable[0] == 0);
    assert(trimed_signal.size(0) >= (int64_t) movetable.back());
    using namespace torch::indexing;
    signal_group.resize(movetable.size() - 1);
    for (size_t move_idx = 0; move_idx < movetable.size() - 1; move_idx++) {
        int64_t start = movetable[move_idx];
        int64_t end = movetable[move_idx + 1];
        signal_group[move_idx] = trimed_signal.index({Slice(start * stride, end * stride)});
    }
}

std::string Yao::get_complement_seq(std::string &ref_seq, std::string seq_type) {
    auto basepairs = Yao::get_basepairs_dna();
    auto basepairs_rna = Yao::get_basepairs_rna();
    std::string ret("");
    if (seq_type == "DNA") {
        for (char &ch: ref_seq) {
            ret = basepairs.at(ch) + ret;
        }
    } else if (seq_type == "RNA") {
        for (char &ch: ref_seq) {
            ret += basepairs_rna.at(ch) + ret;
        }
    } else {
        std::cerr << "seq_type error when get complement seq!\n";
    }
    return ret;
}


char Yao::alphabed(char letter, const std::map<char, char> &basepairs) {
    if (basepairs.find(letter) == basepairs.end()) {
        return 'N';
    }
    return basepairs.at(letter);
}

void Yao::get_cigar_pair(std::vector<std::pair<int32_t, int32_t>> &cigar_pair,
                         const std::string &cigar_str) {
    cigar_pair.clear();
    std::size_t pos = 0;
    for (std::size_t i = 0; i < cigar_str.length(); i++) {
        if (op_set.find(cigar_str[i]) != op_set.end()) {
            std::int32_t op_len = stoi(cigar_str.substr(pos, i - pos));
            std::int32_t op = cigar_convert.at(cigar_str[i]);
            cigar_pair.push_back({op, op_len});
            pos = i + 1;
        }
    }
}

void Yao::parse_cigar(std::vector<std::pair<int32_t, int32_t>> &cigar_pair,
                      std::vector<int32_t> &r_to_q_poss,
                      bool is_forward, int32_t ref_len) {
    r_to_q_poss.clear();
    r_to_q_poss.resize(ref_len + 1, -1);
    int32_t curr_r_pos = 0, curr_q_pos = 0;
    if (!is_forward) std::reverse(cigar_pair.begin(), cigar_pair.end());
    for (const auto &[op, op_len]: cigar_pair) {
        if (op == 1) {
            curr_q_pos += op_len;
        } else if (op == 2 || op == 3) {
            for (int32_t r_pos = curr_r_pos; r_pos < curr_r_pos + op_len; r_pos++) {
                r_to_q_poss[r_pos] = curr_q_pos;
            }
            curr_r_pos += op_len;
        } else if (op == 0 || op == 7 || op == 8) {
            for (int32_t op_offset = 0; op_offset < op_len; op_offset++) {
                r_to_q_poss[curr_r_pos + op_offset] = curr_q_pos + op_offset;
            }
            curr_r_pos += op_len;
            curr_q_pos += op_len;
        } else if (op == 6) {
            // padding (shouldn't happen in mappy)
            continue;
        }
    }
    r_to_q_poss[curr_r_pos] = curr_q_pos;
    if (r_to_q_poss[r_to_q_poss.size() - 1] == -1) {
        std::cerr << "Invalid cigar string encountered, while parsing cigar!\n";
    }
}

int32_t Yao::count_reference_length(std::vector<std::pair<int32_t, int32_t>> &cigar_pair) {
    int32_t ref_len = 0;
    std::set<int32_t> op_set = {0, 2, 3, 7, 8};
    for (const auto &[op, op_len]: cigar_pair) {
        if (op_set.find(op) != op_set.end()) {
            ref_len += op_len;
        }
    }
    return ref_len;
}

float Yao::compute_pct_identity(std::vector<std::pair<int32_t, int32_t>> &cigar_pair, int32_t nm) {
    if (nm == -1) return 0;
    float nmatches = -(float) nm, nalign = 0;
    for (const auto &[op, op_len]: cigar_pair) {
        if (op == 0 || op == 1 || op == 2) {
            nalign += op_len;
        }
        if (op == 0) {
            nmatches += op_len;
        }
    }
    return nmatches / nalign;
}

float Yao::compute_coverage(std::vector<std::pair<int32_t, int32_t>> &cigar_pair, int32_t query_alignment_length) {
    float nmatch = 0;
    for (const auto &[op, op_len]: cigar_pair) {
        if (op == 0) {
            nmatch += op_len;
        }
    }
    if (query_alignment_length == 0.0) {
        return -1;
    }
    return nmatch / (float) query_alignment_length;
}


void Yao::parse_str_by_delimiter(std::vector<std::string> &tokens,
                                 std::string &str,
                                 std::string delimiter) {
    size_t pos = 0;
    std::string token;
    while ((pos = str.find(delimiter)) != std::string::npos) {
        token = str.substr(0, pos);
        str.erase(0, pos + delimiter.length());
        tokens.push_back(token);
    }
    tokens.push_back(str);
}

at::Tensor Yao::get_query_qualities(std::string &query_qualities, bool is_forward) {
    std::vector<float> seq_qualities(query_qualities.length());
    for (size_t i = 0; i < query_qualities.length(); i++) {
        seq_qualities[i] = (float) query_qualities[i] - 33.0;
    }
    if (!is_forward) std::reverse(seq_qualities.begin(), seq_qualities.end());
    at::Tensor ret = torch::from_blob(seq_qualities.data(),
                                      seq_qualities.size(),
                                      torch::kFloat32);
    ret = ret / 60.;
    return ret.clone();
}

at::Tensor Yao::get_signals_rect(std::vector<at::Tensor> &k_signals,
                                 int32_t signals_len) {
    at::Tensor signals_rect = torch::empty({(int64_t) k_signals.size(), signals_len}, torch::kFloat32);
    std::vector<float> vec(signals_len, 0);
    for (size_t i = 0; i < k_signals.size(); i++) {
        vec.resize(signals_len, 0);
        if (k_signals[i].size(0) < signals_len) {
            int32_t pad0_len = signals_len - k_signals[i].size(0);
            int32_t pad0_left = pad0_len / 2;
            for (int64_t j = 0; j < k_signals[i].size(0); j++) {
                vec[j + pad0_left] = k_signals[i][j].item<float>();
            }
        } else if (k_signals[i].size(0) > signals_len) {
            std::vector<int32_t> idxs(k_signals[i].size(0));
            for (int k = 0; k < k_signals[i].size(0); k++) {
                idxs[k] = k;
            }

            std::random_device rd;
            std::mt19937 gen(rd());
            std::shuffle(idxs.begin(), idxs.end(), gen);

            std::vector<int32_t> new_idx(idxs.begin(), idxs.begin() + signals_len);
            std::sort(new_idx.begin(), new_idx.end());
            for (int32_t j = 0; j < signals_len; j++) {
                vec[j] = k_signals[i][new_idx[j]].item<float>();
            }
        } else {
            for (int64_t j = 0; j < k_signals[i].size(0); j++) {
                vec[j] = k_signals[i][j].item<float>();
            }
        }
        signals_rect[i] = torch::from_blob(vec.data(), vec.size(), torch::kFloat32).clone();
        vec.clear();
    }
    return signals_rect;
}

std::map<std::string, int32_t> Yao::get_chr_to_id(const fs::path &path) {
    std::ifstream in(path);
    std::map<std::string, int32_t> map;
    std::string key;
    int32_t value;
    while (in >> key >> value) {
        map[key] = value;
    }
    return map;
}

std::map<std::string, fs::path> Yao::get_filename_to_path(const fs::path &path) {

    std::map<std::string, fs::path> filename_to_path;
    int cnt = 0;
    for (const auto &entry: fs::directory_iterator(path)) {
        if (entry.path().extension() == ".pod5") {
            filename_to_path[entry.path().filename()] = entry.path();
            cnt++;
        }
    }
    spdlog::info("Found {} pod5files", cnt);

    return filename_to_path;
}

std::set<std::string> Yao::get_hc_set(const fs::path &path) {
    std::set<std::string> hc_sites;
    std::ifstream in(path);
    std::string line;
    std::vector<std::string> line_s;
    while (std::getline(in, line)) {
        line_s.clear();
        Yao::parse_str_by_delimiter(line_s, line, "\t");
        if (line_s[0] == "chromosome") continue;
        std::string key = line_s[0] + "#" + line_s[1];
        hc_sites.insert(key);
    }
    spdlog::info("Found {} hc sites from {}", (int64_t) hc_sites.size(), path.filename().string());
    return hc_sites;
}

std::vector<std::string> Yao::generate_keys(at::Tensor &site_info) {
    site_info = site_info.contiguous();
    std::vector<int32_t> T(site_info.data_ptr<int32_t>(), site_info.data_ptr<int32_t>() + site_info.numel());
    std::vector<std::string> keys;
    for (int64_t i = 0; i < site_info.size(0); i++) {
        std::string s1 = std::to_string(T[3 * i]);
        std::string s2 = std::to_string(T[3 * i + 1]);
        std::string s3 = std::to_string(T[3 * i + 2]);
        keys.push_back(s1 + "\t" + s2 + "\t" + s3);
    }
    return keys;
}

//std::string Yao::get_num_to_str(int32_t i)  {
//    static std::stringstream ss;
//    ss.str("");
//    ss << std::setw(10) << std::setfill('0') << i;
//    std::string s = ss.str();
//    try {
//        assert(s.length() == (size_t) 10);
//    }
//    catch (std::runtime_error &e) {
//        spdlog::error("error: {}", e.what());
//        spdlog::error("length of s: {}", s.length());
//        spdlog::error("i: {}, s: {}", i, s);
//    }
//    return s;
//}






