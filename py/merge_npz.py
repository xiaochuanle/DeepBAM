import numpy as np
import os
from tqdm import tqdm

file_path = "/mnt/sde2/20231219_sugarbeet_feature_CG_100x20_02x20/"
file_list = os.listdir(file_path)
file_list = [x for x in file_list if x.endswith(".npz")]
file_list = [file_path + x for x in file_list]
np.random.shuffle(file_list)
print("find {} npz files".format(len(file_list)))

a = 0
kmer = 101
file_size = 10
chunk_size = int((len(file_list) + file_size) / file_size)

merge_dir = "/mnt/sde2/20231219_sugarbeet_feature_CG_100x20_02x20/101k_merge/"

print("Merge every {} npz files into a chunk".format(chunk_size))

while a < len(file_list) - 1:
    file_chunk = file_list[a : a + chunk_size]
    matrix = np.load(file_chunk[0])["features"]
    for f in file_chunk[1:]:
        try:
            matrixx = np.load(f)["features"]
            assert matrixx.shape[1] == 20 * kmer + 1
            matrix = np.append(matrix, matrixx, axis=0)
        except:
            continue
    write_file = merge_dir + "chunk_{}".format(int(a / chunk_size))
    np.random.shuffle(matrix)
    assert matrix.shape[1] == 20 * kmer + 1
    np.save(write_file, matrix)
    chunk_id = int(a / chunk_size)
    print("Writed chunk_{} to file: {}".format(chunk_id, write_file))
    a += chunk_size
    if chunk_id >= 10: break