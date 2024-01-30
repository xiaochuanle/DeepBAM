//
// Created by dell on 2023/8/7.
//

#ifndef BAMCLASS_SAMREAD_H
#define BAMCLASS_SAMREAD_H
#pragma once

#include <iostream>
#include <torch/torch.h>
#include <vector>
#include <map>
#include <set>
#include <unordered_set>

namespace Yao {


    struct SamRead {

        SamRead() = default;

        SamRead(std::string samstr);
//    std::vector<std::pair<std::int32_t ,std::int32_t>> get_ciagr_pair();

        void print() {
//        auto pairs = get_ciagr_pair();
            for (const auto &[a, b]: cigar_pair) {
                std::cout << "(" << a << ", " << b << ") ";
            }
            std::cout << std::endl;
        }

        std::vector<int32_t> get_ref_to_seq();

        std::string get_query_alinment_read();

        std::string get_reference_seq();

        bool is_forward;
        bool is_mapped;
        float map_coverage;
        float map_identity;


        // Basuc alignment section
        std::string query_name;
        int32_t flag;
        std::string reference_name;
        int32_t mapping_quality;
        int32_t reference_start;
        int32_t reference_end;
        int32_t reference_length;
        int32_t query_alignment_start;
        int32_t query_alignment_end;
        int32_t query_alignment_length;
        std::string cigar_string;
        std::vector<std::pair<int32_t, int32_t>> cigar_pair;
        std::string rnext;
        int32_t pnext;
        int64_t tlen;
        std::string query_sequence;
        std::string query_qualities;
        std::string reference_seq;

        // tags
        std::vector<uint64_t> movetable;
        int32_t trimed_start;
        int32_t stride;
        std::string MD_str;
        int32_t NM;
        std::string file_name;
    };

}
#endif //BAMCLASS_SAMREAD_H
