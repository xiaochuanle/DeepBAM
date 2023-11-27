# DeepBam
## Brief Introduction
DeepBam is a model training and inferencing structure especially designed for Oxford Nanopore sequencing data. The model is design with Bi-LSTM. It is trained in python, and use C++ with libtorch for feature extraction and modification calling.

## Build from scrath

### prepare python env
use conda to create vertual environment. Python script requires numpy(>=20.0) and pytorch(>=2.0) with cuda11.8.
```
conda create -n DeepBam python=3.11
conda activate DeepBam
pip install numpy torch==2.0.1
```

### build C++ program
To build this program, you should set up cuda tookit 11.8, and download libtorch (if you have already set up python environment, then libtorch is not necessary). The C++ program is build under Ubuntu 22.04 with g++-11.2, it may raise some problems when build from other systems, if you have these problems don't hesitate to raise a issue.

Before build the program, the following packages should be prepared.

1. boost
2. spdlog
3. zlib

```
git clone https://github.com/huicongyao/Deep-Bam.git
cd Deep-Bam/cpp
mkdir build && cd build
conda activate DeepBam # activate previous created environment
cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)' .. 
make -j
```

## DeepBam usage
After successfully build the program, you could directly use our trained model ,or try to train a model on your own. The executable is Deep-Bam/cpp/build/DeepBam.
### DeepBam: extract hc sites
The extract feature process (for model training).
```
Usage: extract_hc_sites [--help] [--version] pod5_dir bam_path reference_path ref_type write_dir pos neg kmer_size num_workers motif_type loc_in_motif

extract features for model training with high confident bisulfite data

Positional arguments:
  pod5_dir        path to pod5 directory 
  bam_path        path to bam file, sorted by file name is needed 
  reference_path  path to reference genome 
  ref_type        reference genome tyoe [default: "DNA"]
  write_dir       write directory, write file format ${pod5filename}.npy 
  pos             positive high accuracy methylation sites 
  neg             negative high accuracy methylation sites 
  kmer_size       kmer size for extract features [default: 51]
  num_workers     num of workers in extract feature thread pool, every thread contains 4 sub-threads to extract features, so do not choose more than (num of cpu / 4) threads [default: 10]
  motif_type      motif_type default CG [default: "CG"]
  loc_in_motif    Location in motifset 
```

The extracted features is npz file witch contains site info and data info. The site info is a tab-delimited string stored in uint8 array, and the features is the data for training.

### DeepBam: extract and call mods
The call modification process.
```
Usage: extract_and_call_mods [--help] [--version] pod5_dir bam_path reference_path ref_type write_file1 write_file2 module_path kmer_size num_workers batch_size motif_type loc_in_motif

asynchronously extract features and pass data to model to get modification result

Positional arguments:
  pod5_dir        path to pod5 directory 
  bam_path        path to bam file, sorted by file name is needed 
  reference_path  path to reference genome 
  ref_type        reference genome type [default: "DNA"]
  write_file1     write modification count file path 
  write_file2     write detailed modification result file path 
  module_path     module path to trained model 
  kmer_size       kmer size for extract features [default: 51]
  num_workers     num of workers in extract feature thread pool, every thread contains 4 sub-threads to extract features, so do not choose more than (num of cpu / 4) threads [default: 10]
  batch_size      default batch size [default: 4096]
  motif_type      motif_type default CG [default: "CG"]
  loc_in_motif    Location in motifset 
```

The call_mods process outputs two file, one file contains methylation freqency info just like bisulfite data, and the other contains detailed read level methylation infomation from the data you called.