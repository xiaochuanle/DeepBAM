## DeepBam python script specification

This directory contains three python script for model training.

1. `dataloader.py`: a file contains a customized dataloader for different kind of data.

2. `model.py`: model definition.
3. `train_lstm.py`: the python script for model training.

After running DeepBam extract_hc_sites, you need the merge the `npz` file into a large binary file (or a bunch of large binary files if they are two large). The code is currently designed to training data from two different sources. So if you want to train data from a single source, you could change the code for your own needs.

