#!/bin/python

import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.signal import savgol_filter

def process_line(current_data, results):
    for idx, (pos1, prob1) in enumerate(current_data):
        for pos2, prob2 in current_data[idx + 1:]:
            distance = abs(pos1 - pos2)
            prob_status = prob1 + prob2
            results[distance].append(prob_status)

def calculate_correlation(results):
    final_results = defaultdict(dict)
    for distance, prob_statuses in results.items():
        count_0 = prob_statuses.count(0)
        count_1 = prob_statuses.count(1)
        count_2 = prob_statuses.count(2)
        total = len(prob_statuses)
        FM = (count_1 + 2 * count_2 ) / (2 * total) if total > 0 else 0
        FMM = count_2 / total if total > 0 else 0
        correlation = (FMM - FM**2) / (FM * (1 - FM)) if FM * (1 - FM) != 0 else None
        final_results[distance] = {
            'count_0': count_0,
            'Ratio_0': count_0 / total if total > 0 else 0,
            'count_1':count_1,
            'Ratio_1': count_1 / total,
            'count_2': count_2,
            'Ratio_2': FMM,
            'Correlation': correlation
        }
    return final_results

filePath = sys.argv[1]
dist_len = sys.argv[2]

# Initialize variables
results = defaultdict(list)
current_read = None
current_data = []

# Read the file line by line
with open(filePath, 'r') as file:
    next(file)  # Skip header
    for line in file:
        read_name, chrom, pos, prob = line.strip().split('\t')
        pos, prob = int(pos), int(prob)

        if read_name != current_read and current_read is not None:
            process_line(current_data, results)
            current_data = []

        current_read = read_name
        current_data.append((pos, prob))

    # Process the last read
    if current_read is not None:
        process_line(current_data, results)

# Calculate correlation
final_results = calculate_correlation(results)

# Convert results to DataFrame
results_df = pd.DataFrame([(dist, data['count_0'], data['Ratio_0'], data['count_1'], data['Ratio_1'], 
                            data['count_2'], data['Ratio_2'], data['Correlation']) 
                           for dist, data in final_results.items()], 
                          columns=['Distance', 'count_0', 'Ratio_0', 'count_1', 'Ratio_1', 
                                   'count_2', 'Ratio_2', 'Correlation'])

sort_df = results_df.sort_values(by="Distance")

#画图
# df1 是上述代码处理好的数据1
# df2 是上述代码处理好的数据2

plt.figure(figsize=(4,3))

#hg001
x1=np.array(df1.head(dist_len)['Distance'].tolist())
y1=np.array(df1.head(dist_len)['Correlation'].tolist())
y1_smoothed = savgol_filter(y1, window_length=51, polyorder=3)  # 窗口长度和多项式顺序可以调整

#hg002
x2=np.array(df2.head(dist_len)['Distance'].tolist())
y2=np.array(df2.head(dist_len)['Correlation'].tolist())
y2_smoothed = savgol_filter(y2, window_length=51, polyorder=3)  # 窗口长度和多项式顺序可以调整

plt.plot(x1, y1, color='#B8B8B8')
plt.plot(x1, y1_smoothed, color='#FFBE7A', label="HG001")
plt.plot(x2, y2, color='#B8B8B8')
plt.plot(x2, y2_smoothed, color='#FA7F6F', label="HG002")

plt.title("Correlations of methylation states at the single molecule level", fontsize=9)
plt.xlabel("Distance between two CpGs(bp)", fontsize=8)
plt.ylabel("Correlation", fontsize=8)
plt.legend(fontsize=7)

#设置坐标刻度大小
plt.tick_params(axis="both", labelsize=7)
plt.savefig('' , dpi=300, format='svg',bbox_inches='tight')
plt.savefig('', dpi=300, format='png',bbox_inches='tight')
plt.show()




