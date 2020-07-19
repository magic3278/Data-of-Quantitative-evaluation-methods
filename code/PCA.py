from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
pd.set_option('expand_frame_repr', False)

def pca(dataMat, topNfeat=999999):
    meanValues = np.mean(dataMat, axis=0)  # 竖着求平均值，数据格式是m×n
    meanRemoved = dataMat - meanValues  # 0均值化  m×n维
    covMat = np.cov(meanRemoved, rowvar=0)  # 每一列作为一个独立变量求协方差  n×n维
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))  # 求特征值和特征向量  eigVects是n×n维
    eigValInd = np.argsort(-eigVals)  # 特征值由大到小排序，eigValInd十个arrary数组 1×n维
    eigValInd = eigValInd[:topNfeat]  # 选取前topNfeat个特征值的序号  1×r维
    redEigVects = eigVects[:, eigValInd]  # 把符合条件的几列特征筛选出来组成P  n×r维
    lowDDataMat = meanRemoved * redEigVects  # 矩阵点乘筛选的特征向量矩阵  m×r维 公式Y=X*P
    reconMat = (lowDDataMat * redEigVects.T) + meanValues  # 转换新空间的数据  m×n维
    ratio = eigVals / np.sum(eigVals)
    return lowDDataMat, reconMat, redEigVects, eigVals, ratio


path = r'  '
df = pd.read_excel(path,
                  )

df1 = df[['bri', 'car', 'fra', 'perm', 'por']]

# df_nor = df1.apply(lambda x : (x-np.min(x))/(np.max(x)-np.min(x)))
df_nor = df1.apply(lambda x : (x-np.mean(x))/np.std(x))
X = df_nor.values


pca = PCA(n_components=5)

proceed_data = pca.fit_transform(X)
df['1st'] = proceed_data[:,0]
covMat = pca.get_covariance()
eigVals, eigVects = np.linalg.eig(np.mat(covMat))  # 求特征值和特征向量  eigVects是n×n维

print(eigVals)
print(eigVects)



print(pca.explained_variance_)
print(pca.explained_variance_ratio_)

# exit()
df.to_csv('pca.csv')



