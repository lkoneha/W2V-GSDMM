import matplotlib.pyplot as plt
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score
from sklearn.metrics import confusion_matrix, classification_report, davies_bouldin_score, silhouette_score, calinski_harabasz_score
from scipy.stats import mode
from sklearn.cluster import KMeans

import numpy as np
import pandas as pd
f = open ('testLFDMM.theta', 'r')
l = []

l= [line.split() for line in f]


print(l)

truth = np.empty(1076)

np.random.seed(42)
n_samples = 1076
t = 0;
for k in truth:
    if t >= 0 and t < 58:
        truth[t] = 1
        t = t + 1
    if t >= 58 and t < 417:
        truth[t] = 2
        t = t + 1
    if t >= 417 and t < 693:
        truth[t] = 3
        t = t + 1
    if t >= 693 and t < 727:
        truth[t] = 4
        t = t + 1
    if t >= 727 and t < 787:
        truth[t] = 5
        t = t + 1
    if t >= 787 and t < 860:
        truth[t] = 6
        t = t + 1
    if t >= 860 and t < 876:
        truth[t] = 7
        t = t + 1
    if t >= 876 and t < 1036:
        truth[t] = 8
        t = t + 1
    if t >= 1036 and t < 1076:
        truth[t] = 9
        t = t + 1

print(truth)
df2=pd.DataFrame(truth)
df1=pd.DataFrame(l)
print(df1.head())
df2=pd.concat([df1, df2], axis = 1)

#df1.to_csv("tfidf_matrix.csv", header=False, index=False)
    #df1.append(actual.values)
print(df2.head())
df2.to_csv("lfdmm_matrix_DS5.csv", header=False, index=False)
