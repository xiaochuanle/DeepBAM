import numpy as np 
import os

extract_dir = "/public1/YHC/21k/hg002_21k/"
target_dir = "/public1/YHC/13k/hg002/"

ori_kmer = 21
target_kmer = 13

assert(ori_kmer >= target_kmer)

extract_list = [extract_dir + x for x in os.listdir(extract_dir) if x.endswith("npy")]

np.random.shuffle(extract_list)

chunk_id = 0
for p in extract_list[:10]:
    print("loading file: {}".format(p))
    arr = np.load(p)
    kmer = arr[:, :ori_kmer]
    signal = arr[:, ori_kmer:-1].reshape(-1, ori_kmer, 19)
    label = arr[:, -1].reshape(-1, 1)
    
    e_kmer = kmer[:, int((ori_kmer-target_kmer)/2) : int((ori_kmer+target_kmer)/2)]
    e_signal = signal[:, int((ori_kmer-target_kmer)/2) : int((ori_kmer+target_kmer)/2), :]
    e_signal = e_signal.reshape(-1, 19 * target_kmer)
    
    arr_save = np.concatenate((e_kmer, e_signal, label), axis=1)
    assert(arr_save.shape[1] == 20 * target_kmer + 1)
    
    file = target_dir + "chunk_" + str(chunk_id)
    chunk_id += 1
    np.save(file, arr_save)
    print("save {}-kmer chunk to: {}".format(target_kmer, target_dir))