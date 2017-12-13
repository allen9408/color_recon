import numpy as np 
import pandas as pd 
import pdb

df_input = pd.read_csv('input.csv')
X = df_input.as_matrix()

df_label = pd.read_csv('color.csv')
color = df_label.as_matrix()

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
clu_labels = []
sil_score = []
for n in range(2,11):
    labels = KMeans(n_clusters = n).fit_predict(color)
    clu_labels.append(labels)
    silhouette_avg = silhouette_score(color, labels)
    sil_score.append(silhouette_avg)
    print('For n_cluster = ', n, ', score = ', silhouette_avg)
max_n = sil_score.index(max(sil_score))

print(max_n)

colors = ['r','g','b','y','c','m','k','orange','pink','purple']
from sklearn.decomposition import TruncatedSVD
tSNE_model = TSNE(verbose=2, perplexity=30,min_grad_norm=1E-12,n_iter=3000)
z_run_tsne = tSNE_model.fit_transform(color)
f1, ax1 = plt.subplots(1, 1)
ax1.scatter(z_run_tsne[:,0],z_run_tsne[:,1], c = [colors[i] for i in clu_labels[max_n]], marker='*',linewidths = 0)
ax1.set_title('tSNE')
f1.savefig('tsne.png')
f1.clf()
plt.close('all')
print(clu_labels[max_n])