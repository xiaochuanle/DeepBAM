//
// Created by dell on 2023/8/7.
//

#include <cassert>
#include "SamRead.h"
#include "../utils/utils_func.h"


Yao::SamRead::SamRead(std::string samstr){
    std::string delimiter = "\t";
    std::vector<std::string> tokens;
    Yao::parse_str_by_delimiter(tokens, samstr, "\t");
    // get base info
    query_name = tokens[0];
    flag = stoi(tokens[1]);
    reference_name = tokens[2];
    reference_start = stoi(tokens[3]) - 1; // 1 based in bam file, 0 based in pysam
    mapping_quality  = stoi(tokens[4]);
    cigar_string = tokens[5];
    rnext = tokens[6];
    pnext = stoi(tokens[7]);
    tlen = stoi(tokens[8]);
    if (tokens[9] != "*") query_sequence = tokens[9];
    if (tokens[10] != "*") query_qualities  = tokens[10];
    is_forward = !(flag & 0x10);
    is_mapped = !(flag & 0x4);
    file_name = "";
    trimed_start = -1;
    stride = -1;

    // get tags needed
    for (size_t i = 11; i < tokens.size(); i++) {
        if (tokens[i].substr(0, 5) == "fn:Z:") {
            file_name = tokens[i].substr(5);
        }else if (tokens[i].substr(0, 5) == "ts:i:") {
            trimed_start = stoi(tokens[i].substr(5));
        }else if (tokens[i].substr(0, 6) == "mv:B:c") {
            stride = tokens[i][7] - 48;
            Yao::get_movetable(movetable, tokens[i]);
        }else if (tokens[i].substr(0, 5) == "NM:i:") {
            NM = stoi(tokens[i].substr(5));
        }
        else if (tokens[i].substr(0, 5) == "MD:Z:") {
            MD_str = tokens[i].substr(5);
        }
    }

    if (is_mapped){
        Yao::get_cigar_pair(cigar_pair, cigar_string);

        //
        reference_length = Yao::count_reference_length(cigar_pair);
        reference_end = reference_start + reference_length;

        // get query info
        if (cigar_pair[0].first == 4) {
            query_alignment_start = cigar_pair[0].second;
        } else query_alignment_start = 0;
        if (cigar_pair.back().first == 4)
            query_alignment_end = query_sequence.length() - cigar_pair.back().second;
        else query_alignment_end = query_sequence.length();
        query_alignment_length = query_alignment_end - query_alignment_start;
    }else {
        reference_start = -1;
        reference_end = -1;
        reference_length = -1;
        query_alignment_start = -1;
        query_alignment_end = -1;
        query_alignment_length = -1;
    }
}
