#!/bin/python

import sys
import pandas as pd
import re
import subprocess
from Bio import SeqIO
import argparse

def read_file_to_dict(file_path, t):
    result_dict = {}
    count_dict = {}  # 存储每个键的信息
    with open(file_path, 'r') as file:
        # 读取并忽略表头
        header = file.readline()

        for line in file:
            cols = line.strip().split('\t')  # 假设列是通过制表符分隔的
            key = (cols[3], int(cols[4]))  # 将位置信息作为元组，设为键
            value = float(cols[6])  # methlation_rate作为值

            me = 1 if value > t else 0

            if key in result_dict:
                result_dict[key].append(me)
                count_dict[key]['count'] += 1
                count_dict[key]['methy'] += me
            else:
                result_dict[key] = [me]
                count_dict[key] = {'count': 1, 'methy': me}

    return result_dict, count_dict

def filter_data(file_path, t):
    df = pd.read_csv(file_path, sep='\t')
    df_filtered = df[(df.iloc[:, 6] < 1-t) | (df.iloc[:, 6] >= t)]
    
    return df_filtered

def write_cg_motifs_to_bed(fasta_file):
    bed_file = f"{fasta_file}.cg_motifs.bed"
    with open(bed_file, "w") as f:
        for record in SeqIO.parse(fasta_file, "fasta"):
            seq = str(record.seq).upper()
            cg_motifs = [(match.start(), match.end()) for match in re.finditer(r"CG", seq)]
            if cg_motifs:
                for motif in cg_motifs:
                    combined_col = f"{record.id}:{motif[0]}"
                    f.write(f"{record.id}\t{motif[0]}\t{motif[1]}\t{combined_col}\n")
    return bed_file

def preprocess_summary_file(summary_file):
    # Load the summary file
    df = pd.read_csv(summary_file, sep='\t')
    
    # Remove the header and save to a temporary file
    temp_file = "temp_summary_no_header.txt"
    df.to_csv(temp_file, sep='\t', index=False, header=False)
    
    # Sort the file
    sorted_file = "sorted_summary_file.txt"
    sort_command = f"sort -k1,1 -k2,2n {temp_file} > {sorted_file}"
    subprocess.run(sort_command, shell=True, check=True)
    
    return sorted_file

def intersect_bed_files(bed_file, summary_file, output_file, awk_script):
    sorted_summary_file = preprocess_summary_file(summary_file)
    
    # Use bedtools to intersect and awk to process the output
    intersect_command = f"bedtools intersect -wo -a {bed_file} -b {sorted_summary_file} | awk -f {awk_script} > {output_file}"
    subprocess.run(intersect_command, shell=True, check=True)

def process_fasta_and_intersect(fasta_file, summary_file, awk_script):
    # Step 1: Write CG motifs to BED file
    bed_file = write_cg_motifs_to_bed(fasta_file)

    # Step 2: Intersect BED files and process the output
    output_file = f"{summary_file}.mergeStrand.cov"
    intersect_bed_files(bed_file, summary_file, output_file, awk_script)

    return output_file

def main():
    parser = argparse.ArgumentParser(description="Process methylation data and optionally merge strand information.")
    parser.add_argument('--file_path', required=True, help='Path to the input file.')
    parser.add_argument('--threshold', required=True, type=float, help='Threshold for methlation_rate.')
    parser.add_argument('--aggregation_output', help='File name to save summary_filtered DataFrame.')
    parser.add_argument('--filter_ratio', type=float, help='Proportion of rows to filter out based on uncertainty_interval (e.g., 0.1 for 10%).')
    parser.add_argument('--merge_strand', help='If provided, performs strand merging using the specified fasta file.')
    parser.add_argument('--awk_script', default='process.awk', help='Path to the AWK script file.')

    args = parser.parse_args()

    # 处理筛选数据
    if args.filter_ratio is not None:
        filtered_df = filter_data(args.file_path, args.filter_ratio)
        temp_filtered_file = "filtered_temp_file.txt"
        filtered_df.to_csv(temp_filtered_file, sep='\t', index=False)
        data_dict, count_dict = read_file_to_dict(temp_filtered_file, args.threshold)
    else:
        data_dict, count_dict = read_file_to_dict(args.file_path, args.threshold)

    summary_data = []
    for key, valist in data_dict.items():
        meratio = sum(valist) / len(valist)
        end = key[1] + 1
        unmethy = count_dict[key]['count'] - count_dict[key]['methy']
        summary_data.append([key[0], key[1], end, meratio * 100, count_dict[key]['methy'], unmethy])
    summary_df = pd.DataFrame(summary_data, columns=['chr', 'start', 'end', 'frequency', 'methy', 'unmethy'])

    summary_output = args.aggregation_output if args.aggregation_output else "detail_aggregation.txt"
    summary_df.to_csv(summary_output, sep='\t', index=False)

    # 检查是否需要执行 strand merging
    if args.merge_strand:
        process_fasta_and_intersect(args.merge_strand, summary_output, args.awk_script)

if __name__ == "__main__":
    main()

