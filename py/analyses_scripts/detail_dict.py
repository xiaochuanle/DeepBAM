#!/bin/python

import sys
import pandas as pd
import time

def read_file_to_dict(file_path, t):
    result_dict = {}
    count_dict = {} 
    with open(file_path, 'r') as file:
        header = file.readline()
        for line in file:
            cols = line.strip().split('\t')  
            key = (cols[3], int(cols[4]) )  
            value = float(cols[6])
            
            me = 1 if value > t else 0

            if key in result_dict:
                result_dict[key].append(me)
                count_dict[key]['count'] += 1
                count_dict[key]['methy'] += me
            else:
                result_dict[key] = [me]
                count_dict[key] = {'count':1, 'methy':me}

    return result_dict, count_dict

file_path = sys.argv[1]
t = float(sys.argv[2])
wgbs = sys.argv[3]
output = sys.argv[4]

data_dict, count_dict= read_file_to_dict(file_path, t)

summary_data = []
for key, valist in data_dict.items():
    meratio = sum(valist) / len(valist)
    summary_data.append([key[0], key[1], meratio * 100, count_dict[key]['methy'], count_dict[key]['count']])
summary_df = pd.DataFrame(summary_data, columns=['chr', 'start', 'frequency', 'methy', 'count'])
summary_filtered = summary_df[summary_df['count'] > 5]


wgbs = pd.read_csv(wgbs,
                  sep='\t', index_col=False,
                   names=['chr', 'start', 'end', 'ratio', 'cov'])
wgbs = wgbs.dropna(axis=0)
wgbs_filtered = wgbs[wgbs['cov'] > 5]

inter_df = pd.merge(summary_filtered, wgbs_filtered,
                    on=['chr', 'start'])

print(inter_df[['ratio', 'frequency']].corr())

inter_df[['ratio', 'frequency']].to_csv(output, sep='\t', index=None)
