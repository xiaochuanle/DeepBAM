import sys
import pandas as pd
import numpy as np
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score

file1 = sys.argv[1]
file2 = sys.argv[2]
pos = int(sys.argv[3])
neg = int(sys.argv[4])
cores = int(sys.argv[5])
output = sys.argv[6]


wgbs = pd.read_csv(file1, sep='\t',names=['chr', 'start', 'end', 'ratio', 'cov'])

#for modbam2bed file
ont = pd.read_csv(file2, sep='\t', index_col=False
        names=['read_id','ref_start', 'ref_end', 'chr', 'start', 'strand', 'prob']
        )


inter_df = pd.merge(wgbs, ont, on=['chr','start'])

# label high confidence
t_df = inter_df.loc[(inter_df['ratio'] >= pos) | (inter_df['ratio'] <= neg)]
t_df['label'] = t_df.apply(lambda row: 1 if row['ratio'] > 0.5 else 0, axis=1)

#calculate AUC and print
auc = roc_auc_score(t_df['label'], t_df['prob'])
ap = average_precision_score(t_df['label'], t_df['prob'])
print(f"AUC value of ROC is: {auc:4f}")
print(f"AP value of PR is: {ap:4f}")

#define function to tranform predict to binary classification
def binarize(pred, threshold):
    if pred >= threshold:
        return 1
    else:
        return 0
threshold = [i/100 for i in range(101)]

#loop the threshold list, calculate and write into result
def ROC(t, df=t_df.copy()):
    #apply binarize function
    df['bin'] = df['prob'].apply(binarize, args=(t, ))
    # calculate
    TP = ((df["bin"] == 1) & (df["label"] == 1)).sum()
    TN = ((df["bin"] == 0) & (df["label"] == 0)).sum()
    FP = ((df["bin"] == 1) & (df["label"] == 0)).sum()
    FN = ((df["bin"] == 0) & (df["label"] == 1)).sum()
    TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
    F1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0
    Precision = TP / (TP+FP) if (TP+FP) != 0 else 0
    Accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
    Recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    Specificity = TN / (TN + FN) if (TN + FN) != 0 else 0
    #write into result
    return [t, TP, TN, FP, FN, TPR, FPR, F1, Precision, Accuracy, Recall, Specificity]

#output results
p = Pool(cores)
results = p.map(ROC, threshold)
results_df = pd.DataFrame(results, columns=["threshold", 'TP', 'TN', 'FP',
                                            'FN', "TPR", "FPR", "F1",
                                            "Precision", "Accuracy", "Recall", "Specificity"])
results_df.to_csv(f"{output}", sep='\t', index=False)


