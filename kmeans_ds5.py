import pandas as pd
import re, nltk, gensim
import spacy

from pprint import pprint
import gensim
import nltk
import re
from pprint import pprint
import numpy as np
import pandas as pd
import spacy
import csv
import multiprocessing
from gensim.models import Word2Vec
from nltk.cluster import KMeansClusterer
from sklearn import cluster
from sklearn import metrics
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score
from sklearn.metrics import confusion_matrix, classification_report, davies_bouldin_score, silhouette_score, calinski_harabasz_score
from scipy.stats import mode
from sklearn.cluster import KMeans


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
        truth[t] = 0
        t = t + 1
    if t >= 58 and t < 417:
        truth[t] = 1
        t = t + 1
    if t >= 417 and t < 693:
        truth[t] = 2
        t = t + 1
    if t >= 693 and t < 727:
        truth[t] = 3
        t = t + 1
    if t >= 727 and t < 787:
        truth[t] = 4
        t = t + 1
    if t >= 787 and t < 860:
        truth[t] = 5
        t = t + 1
    if t >= 860 and t < 876:
        truth[t] = 6
        t = t + 1
    if t >= 876 and t < 1036:
        truth[t] = 7
        t = t + 1
    if t >= 1036 and t < 1076:
        truth[t] = 8
        t = t + 1

print(truth)
K = range(1, 35)

n_clusters=9
sg_kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0, max_iter=2000).fit(l)
clusters = sg_kmeans.labels_
sg_cluster_centroids = sg_kmeans.cluster_centers_
kmeans1_ = clusters.tolist()
#actual1 = actual.tolist()
K = range(2, 15)
csvfile1 = "./sc5.csv"
scc = []
for k in K:
    km = KMeans(n_clusters=k, init='k-means++')
    km = km.fit(l)
    clusters = km.labels_
    j = silhouette_score(l, clusters, metric='euclidean')
    csvrow1 = [k, j]
    if k ==3 or k==6 or k==9 or k==12 or k==15:
        with open(csvfile1, "a", newline='') as fp:
            wr = csv.writer(fp, dialect='excel')
            wr.writerow(csvrow1)
    scc.append(silhouette_score(l, clusters, metric='euclidean'))
    #    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, scc, 'bx-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette_score')
    # plt.title('Elbow Method For Optimal k')
plt.show()

K = range(1, 25)

print(clusters)
c = Counter(clusters)
print(c.items())
k_labels = clusters
k_labels_matched = np.empty_like(clusters)
print(k_labels.shape)

kmeans1_ = clusters.tolist()
#actual1 = actual.tolist()

print("kmeans: %.4f" % normalized_mutual_info_score(truth, kmeans1_))
print("kmeans: %.4f" % adjusted_mutual_info_score(truth, kmeans1_))
print("kmeans: %.4f" % adjusted_rand_score(truth, kmeans1_))
print("kmeans: %.4f" % homogeneity_score(truth, kmeans1_))
print("kmeans: %.4f" % completeness_score(truth, kmeans1_))
print("kmeans: %.4f" % v_measure_score(truth, kmeans1_))
nmi = normalized_mutual_info_score(truth, kmeans1_)
ami = adjusted_mutual_info_score(truth, kmeans1_)
ari = adjusted_rand_score(truth, kmeans1_)
hm = homogeneity_score(truth, kmeans1_)
com = completeness_score(truth, kmeans1_)
vm = v_measure_score(truth, kmeans1_)

    # print(actual.shape)

for i in range(9):
        mask = (clusters == i)
        k_labels_matched[mask] = mode(truth[mask])[0]

print(classification_report(truth, k_labels_matched))
print(confusion_matrix(truth, k_labels_matched))
print(accuracy_score(truth, k_labels_matched))
print(silhouette_score(l, clusters, metric='euclidean'))
#print(calinski_harabasz_score(l, clusters))
#print(davies_bouldin_score(l, clusters))

ac = accuracy_score(truth, k_labels_matched)
#db = davies_bouldin_score(l, clusters)
sc = silhouette_score(l, clusters, metric='euclidean')
#ch = calinski_harabasz_score(l, clusters)

csvrow = [nmi, ami, ari, hm, com, vm, ac, sc]
csvfile = "./re.csv"
with open(csvfile, "a", newline='') as fp:
    wr = csv.writer(fp, dialect='excel')
    wr.writerow(csvrow)