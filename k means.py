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

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
from gensim.models import KeyedVectors
from gensim.models import word2vec
corpus_root = 'C:/Users/Saurabh Jaiswal/PycharmProjects/New_for graphs/DS1_RS4/Dataset1_PW/DS4_WELDA.csv'
df = pd.read_csv(corpus_root)
df_desc = df['Description']
actual = df['Categoryl']
f = open ('testLFLDMM.theta', 'r')
l = []

l= [line.split() for line in f]


print(l)
K = range(2, 30)
csvfile1 = "./sc4.csv"
scc = []
for k in K:
    km = KMeans(n_clusters=k, init='k-means++')
    km = km.fit(l)
    clusters = km.labels_
    j = silhouette_score(l, clusters, metric='euclidean')
    csvrow1 = [k, j]
    if k ==5 or k==10 or k==15 or k==20 or k==25:
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

n_clusters=20
sg_kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0, max_iter=2000).fit(l)
clusters = sg_kmeans.labels_
sg_cluster_centroids = sg_kmeans.cluster_centers_
kmeans1_ = clusters.tolist()
actual1 = actual.tolist()

print(clusters)
c = Counter(clusters)
print(c.items())
k_labels = clusters
print("clusters")
print(k_labels)
k_labels_matched = np.empty_like(clusters)
print(k_labels.shape)

kmeans1_ = clusters.tolist()
actual1 = actual.tolist()

print("kmeans: %.4f" % normalized_mutual_info_score(actual1, kmeans1_))
print("kmeans: %.4f" % adjusted_mutual_info_score(actual1, kmeans1_))
print("kmeans: %.4f" % adjusted_rand_score(actual1, kmeans1_))
print("kmeans: %.4f" % homogeneity_score(actual1, kmeans1_))
print("kmeans: %.4f" % completeness_score(actual1, kmeans1_))
print("kmeans: %.4f" % v_measure_score(actual1, kmeans1_))
nmi = normalized_mutual_info_score(actual1, kmeans1_)
ami = adjusted_mutual_info_score(actual1, kmeans1_)
ari = adjusted_rand_score(actual1, kmeans1_)
hm = homogeneity_score(actual1, kmeans1_)
com = completeness_score(actual1, kmeans1_)
vm = v_measure_score(actual1, kmeans1_)
print(type(clusters))
    # print(actual.shape)
print(len(clusters))
for i in range(20):
    mask = (clusters == i)
    print(mask)
    k_labels_matched[mask] = mode(actual[mask])[0]

print(classification_report(actual, k_labels_matched))
print(confusion_matrix(actual, k_labels_matched))
print(accuracy_score(actual, k_labels_matched))
print(silhouette_score(l, clusters, metric='euclidean'))
#print(calinski_harabasz_score(l, clusters))
#print(davies_bouldin_score(l, clusters))

ac = accuracy_score(actual, k_labels_matched)
#db = davies_bouldin_score(l, clusters)
sc = silhouette_score(l, clusters, metric='euclidean')
#ch = calinski_harabasz_score(l, clusters)

csvrow = [nmi, ami, ari, hm, com, vm, ac, sc]
csvfile = "./re.csv"
with open(csvfile, "a", newline='') as fp:
    wr = csv.writer(fp, dialect='excel')
    wr.writerow(csvrow)