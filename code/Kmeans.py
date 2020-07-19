import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
pd.set_option('expand_frame_repr', False)

path = r'  '
df = pd.read_excel(path,
                  )

# df_nor = df.apply(lambda x : (x-np.min(x))/(np.max(x)-np.min(x)))
df1 = df[['bri', 'car', 'fra', 'perm', 'por']]
df_nor = df1.apply(lambda x : (x-np.mean(x))/np.std(x))
X = df_nor.values


# model = KMeans(n_clusters = 3, max_iter = 300).fit(X)
model = KMeans(init='k-means++', n_clusters = 4, n_init=10).fit(X)
# model = MiniBatchKMeans(init='k-means++', n_clusters=3, batch_size=500, n_init=10, max_no_improvement=10, verbose=0).fit(X)
y_pred = model.labels_

df['label'] = y_pred
print(df)
df.to_csv('after_classify_100.csv')
