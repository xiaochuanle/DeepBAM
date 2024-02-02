#!/bin/bash

THREADS=$1
INPUT=$2
COV=$3

for coverage in 5 10 15 20 25 30
do
  total_coverage=${COV}
  downsample_fraction=0.$((coverage * 100 / total_coverage))
  echo "Coverage= ${coverage}, Downsample fraction = ${downsample_fraction}"

  samtools view -s $downsample_fraction -b -@${THREADS} ${INPUT} > ${coverage}x.bam
  samtools index -@${THREADS} ${coverage}x.bam
done

