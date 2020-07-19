import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

pd.set_option('expand_frame_repr', False)

path = r'  '
df = pd.read_excel(path,
                  )
df = df[['bri', 'car', 'fra', 'perm', 'por']]
# df = df.apply(lambda x : (x-np.min(x))/(np.max(x)-np.min(x)))
df = df.apply(lambda x : (x-np.mean(x))/np.std(x))

X = df.values

plt.subplot(1, 2, 1)
SSE = []
for k in range(2, 15):
    estimator = KMeans(n_clusters=k)
    estimator.fit(X)
    SSE.append(estimator.inertia_)
x = range(2, 15)
plt.title('elbow method')
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(x, SSE, 'o-')

plt.subplot(1, 2, 2)
silhouettescore=[]
for i in range(2,15):
    kmeans = KMeans(n_clusters=i).fit(X)
    score=silhouette_score(X,kmeans.labels_)
    silhouettescore.append(score)
plt.title('silhouette method')
plt.xlabel('k')
plt.ylabel('silhouette_score')
plt.plot(range(2,15),silhouettescore, 'o-')

plt.show()