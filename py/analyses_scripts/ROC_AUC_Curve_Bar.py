import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score

def calculate_adjusted_values(df):
    # 确保传入的df是DataFrame类型
    if not isinstance(df, pd.DataFrame):
        raise ValueError('输入必须是一个pandas DataFrame。')
    
    # 复制DataFrame以避免修改原始数据
    df_adjusted = df.copy()
    
    # 计算TP+FN和TN+FP
    df_adjusted['TP_FN'] = df_adjusted['TP'] + df_adjusted['FN']
    df_adjusted['TN_FP'] = df_adjusted['TN'] + df_adjusted['FP']
    
    # 计算倍数，确保不会除以零
    df_adjusted['multiplier'] = df_adjusted['TP_FN'] / df_adjusted['TN_FP'] if df_adjusted['TN_FP'].all() else 0
    
    # 计算乘以倍数后的TN和FP值，并四舍五入转换为整数，计算平衡后的数据
    df_adjusted['adjusted_TN'] = (df_adjusted['TN'] * df_adjusted['multiplier']).round().astype(int)
    df_adjusted['adjusted_FP'] = (df_adjusted['FP'] * df_adjusted['multiplier']).round().astype(int)
    df_adjusted['adjusted_TP'] = (df_adjusted['TP'] * df_adjusted['multiplier']).round().astype(int)
    df_adjusted['adjusted_FN'] = (df_adjusted['FN'] * df_adjusted['multiplier']).round().astype(int)
    
    # 计算调整后的Precision和Specificity
    df_adjusted['b_Precision'] = df_adjusted['TP'] / (df_adjusted['TP'] + df_adjusted['adjusted_FP'])
    df_adjusted['b_Specificity'] = df_adjusted['adjusted_TN'] / (df_adjusted['adjusted_TN'] + df_adjusted['FN'])
    
    # 返回调整后的DataFrame
    return df_adjusted

def filter_data(df, lower, upper):
    return df[(df['threshold'] > lower) & (df['threshold'] < upper)]

def calculate_high_confidence_ratio(df1, df2, thresholds):
    final_results = []
    for threshold in thresholds:
        # Calculate total count from the first row for df1 and df2
        total_df1 = df1.iloc[0][['TP', 'FP', 'TN', 'FN']].sum()
        total_df2 = df2.iloc[0][['TP', 'FP', 'TN', 'FN']].sum()

        # Filter for Precision >= threshold for both dataframes
        precision_filtered_df1 = df1[df1['b_Precision'] >= threshold]
        precision_filtered_df2 = df2[df2['b_Precision'] >= threshold]

        # Filter for Specificity >= threshold for both dataframes
        specificity_filtered_df1 = df1[df1['b_Specificity'] >= threshold]
        specificity_filtered_df2 = df2[df2['b_Specificity'] >= threshold]

        # Initialize variables to 0
        TP1 = TP2 = TP3 = TP4 = 0
        TN1 = TN2 = TN3 = TN4 = 0

        # Process df1 if at least one of the conditions is met
        if not precision_filtered_df1.empty:
            TP1 = precision_filtered_df1.iloc[0]['TP']
            TN1 = precision_filtered_df1.iloc[0]['TN']

        if not specificity_filtered_df1.empty:
            TP2 = specificity_filtered_df1.iloc[-1]['TP']
            TN2 = specificity_filtered_df1.iloc[-1]['TN']

        high_confidence_ratio_df1 = (TP1 + TN2) / total_df1 if total_df1 else None

        # Process df2 if at least one of the conditions is met
        if not precision_filtered_df2.empty:
            TP3 = precision_filtered_df2.iloc[0]['TP']
            TN3 = precision_filtered_df2.iloc[0]['TN']

        if not specificity_filtered_df2.empty:
            TP4 = specificity_filtered_df2.iloc[-1]['TP']
            TN4 = specificity_filtered_df2.iloc[-1]['TN']

        high_confidence_ratio_df2 = (TP3 + TN4) / total_df2 if total_df2 else None
        
        final_results.append({
            'Threshold': threshold,
            'total_df1': total_df1,
            'total_df2': total_df2,
            'High Confidence Ratio DF1': high_confidence_ratio_df1,
            'High Confidence Ratio DF2': high_confidence_ratio_df2,
            'TP1': TP1, 'TN1': TN1, 
            'TP2': TP2, 'TN2': TN2, 
            'TP3': TP3, 'TN3': TN3, 
            'TP4': TP4, 'TN4': TN4
        })

    # Convert to DataFrame
    return pd.DataFrame(final_results)


dpf1 = pd.read_csv("/data3/baixin/arabidopsis/validation/deepbam.F1_AUC.txt", sep='\t', index_col=False)
ddf1 = pd.read_csv("/data3/baixin/arabidopsis/validation/dorado.F1_AUC.txt", sep='\t', index_col=False)

# arabidopsis
sdp1 = filter_data(calculate_adjusted_values(dpf1), 0, 0.5)
pdp1 = filter_data(calculate_adjusted_values(dpf1), 0.5, 1)
sdd1 = filter_data(calculate_adjusted_values(ddf1), 0, 0.5)
pdd1 = filter_data(calculate_adjusted_values(ddf1), 0.5, 1)

#Threshold-True_Calls_Ratio
fig, ax1 = plt.subplots(figsize=(4, 4))

# Plotting the first subplot
ax1.plot(sdp1['threshold'], sdp1['b_Specificity'], color='#66C2A5', label='DeepBam')
ax1.plot(pdp1['threshold'], pdp1['b_Precision'], color='#66C2A5')
ax1.plot(sdd1['threshold'], sdd1['b_Specificity'], color='#FC8D62', label='Dorado')
ax1.plot(pdd1['threshold'], pdd1['b_Precision'], color='#FC8D62')
ax1.set_xlabel('Threshold', fontsize=8)
ax1.set_ylabel('Precision', fontsize=8)
ax1.axvline(x=0.5, color='gray', linestyle='--')
ax1.legend(fontsize=7)
  
plt.title('A.thaliana', fontstyle='italic', fontsize=9)

plt.savefig('', dpi=300, format='svg',bbox_inches='tight')
plt.savefig('', dpi=300, format='png',bbox_inches='tight')

plt.show()

thresholds = np.arange(0.95,0.999,0.01)
final_df1 = calculate_high_confidence_ratio(calculate_adjusted_values(dpf1), calculate_adjusted_values(ddf1), thresholds)
final_df1 = final_df1.fillna(0)

# 设置柱状图的宽度
bar_width = 0.3
# Create a figure with two subplots
fig, ax1 = plt.subplots(figsize=(4, 4))

# 设置x轴的位置
index = np.arange(len(final_df1['Threshold']))

# 绘制第一个柱状图（ratio1）
plt.bar(index, final_df1['High Confidence Ratio DF1'], bar_width, label='DeepBAM', color='#66C2A5')

# 绘制第二个柱状图（ratio2）
plt.bar(index + bar_width, final_df1['High Confidence Ratio DF2'], bar_width, label='Dorado', color='#FC8D62')

# 设置x轴的标签
plt.xlabel('Precision')
# 设置y轴的标签
plt.ylabel('High Confidence Calls Ratio')

# 设置x轴刻度标签
plt.xticks(index + bar_width / 2, final_df1['Threshold'])
plt.ylim(0,1)

# 添加图例
plt.legend(fontsize=7)  
plt.title('A.thaliana', fontstyle='italic', fontsize=9)

# 显示图表
plt.tight_layout()

plt.savefig('', dpi=300, format='svg',bbox_inches='tight')
plt.savefig('', dpi=300, format='png',bbox_inches='tight')

plt.show()
