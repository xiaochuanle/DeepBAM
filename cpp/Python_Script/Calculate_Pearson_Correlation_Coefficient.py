import numpy as np
import torch
from torchmetrics.regression import PearsonCorrCoef
import time

st = time.time()
pearson = PearsonCorrCoef()
bisulfite_path = "/public2/YHC/YHC/Modification_Result/HG002_bisulfite_all_chr.cov"

bisulfite_dict = {}
cnt = 0
for line in open(bisulfite_path):
    line_s = line.strip().split("\t")
    if int(line_s[6]) <= 5: continue
    line_key = line_s[0] + "#" + line_s[1] + "#" + line_s[2]
    bisulfite_dict[line_key] = float(line_s[8])


site_dict = {}

cnt = 0
for line in open("/tmp/data/mod_result.txt"):
    line_s = line.split("\t")
    # break
    if line_s[0] == "Chr_id": continue
    if int(line_s[4]) <= 5: continue
    # if chr != "chr8": continue
    site_dict[line_s[0] + "#" + line_s[1] + "#" + line_s[2]] = float(line_s[3])


x, y = [], []
for site in site_dict.keys():
    if site in bisulfite_dict.keys():
        if site.split("#")[0] != "chr8": continue
        x.append(site_dict[site])
        y.append(bisulfite_dict[site])
x, y = torch.tensor(x), torch.tensor(y)
coef = pearson(x, y)
ed = time.time()
print("Calculated Pearson Correlation coef: {}, total time cost: {} seconds".format(coef, ed - st))