# DeepBam
## Brief Introduction
DeepBam is a model training and inferencing structure especially designed for Oxford Nanopore sequencing data. The model is design with Bi-LSTM. It is trained in python, and use C++ with libtorch for feature extraction and modification calling.

## Build from scrath
### build C++ program
```
git clone https://github.com/huicongyao/Deep-Bam.git
cd Deep-Bam/cpp
mkdir build
cmake ..
make -j
```

### prepare python env
use conda to create vertual environment. Python script requires numpy(>=20.0) and pytorch(>=2.0) with cuda11.8.
```
conda create -n DeepBam python=3.11
conda activate DeepBam
pip install numpy torch==2.0.1
```
