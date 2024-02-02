import numpy as np
import os
import random 

hg002 = "/public1/YHC/21k/hg002_21k/"
sugb = "/public1/YHC/21k/sugarbeet_21k_merge/"
merge = "/public1/YHC/21k/merge_hg002_and_sugb/"

hg002_list = [hg002 + x for x in os.listdir(hg002)]
sugb_list = [sugb + x for x in os.listdir(sugb)]

random.shuffle(hg002_list)
random.shuffle(sugb_list)

print("merge 21k hg002 and sugarbeet")

for i in range(len(sugb_list)):
    arr1 = np.load(hg002_list[i])
    arr2 = np.load(sugb_list[i])
    arr3 = np.concatenate((arr1, arr2), axis=0)
    np.random.shuffle(arr3)
    file_save = merge + "merge_" + str(i)
    np.save(file_save, arr3)
    print("write file to {}".format(file_save))

