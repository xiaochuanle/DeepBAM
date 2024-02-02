#!/bin/bash

# 步骤 1: 输入 detail 文件和测序深度
THREADS=$1
INPUT_BAM=$2
DETAIL_FILE=$3
COV=$4
WGBS=$5
MTHRES=$6

# 步骤 1: 使用 downSample.sh 生成不同深度的 downsampled BAM 文件
echo downSample
bash downSample.sh $THREADS $INPUT_BAM $COV

# 步骤 2: 从每个 downsampled BAM 文件中提取 read names
for coverage in 5 10 15 20 25 30; do
    echo get ${coverage}x readnames
    samtools view ${coverage}x.bam | cut -f1 | sort -u > ${coverage}x_readnames.txt
done

# 步骤 3: detail_chunk.py 提取 detail 文件中的相应 read names 行
for coverage in 5 10 15 20 25 30; do
    echo ${coverage}x_detail.correlation
    grep -F -f ${coverage}x_readnames.txt $DETAIL_FILE > ${coverage}x_detail.txt
    sort -k4,4 -k5,5n ${coverage}x_detail.txt > ${coverage}x_detail.sort.txt
    python detail_dict.py ${coverage}x_detail.sort.txt ${MTHRES} ${WGBS} ${coverage}x_detail.intersectWGBS.txt
    rm ${coverage}x_detail.txt
done


