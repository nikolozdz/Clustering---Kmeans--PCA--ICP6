import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")




# read Data
data = pd.read_csv('CC.csv')




print('Before Cleaning nulls')
nulls = pd.DataFrame(data.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)
#Handling Null values
cleanData = data.select_dtypes(include=[np.number]).interpolate().dropna()
print('After Cleaning nulls')
nulls = pd.DataFrame(cleanData.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)




xTrain = cleanData.iloc[:,[2,-5,-6]]


scaler = preprocessing.StandardScaler()
scaler.fit(xTrain)
xScaledArray = scaler.transform(xTrain)
xScaled = pd.DataFrame(xScaledArray, columns = xTrain.columns)

wcss = []

#11111111111111111111111111111111111111111111111111111111111111111111111111111111
#elbow method to figure out the number of clusters
for i in range(2,12):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(xTrain)
    wcss.append(kmeans.inertia_)
    #222222222222222222222222222222222222222222222222222222222222222222222222222
    score = silhouette_score(xTrain, kmeans.labels_, metric='euclidean')
    print("for n_cluster = {}, silhouette score = {})".format(i, score))



plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()


# from sklearn import metrics
wcss = []
#3333333333333333333333333333333333333333333333333333333333333333333333333
#elbow method to figure out the number of clusters
for i in range(2,12):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(xScaled)
    wcss.append(kmeans.inertia_)
    score = silhouette_score(xScaled, kmeans.labels_, metric='euclidean')
    print("For n_clusters = {}, silhouette score = {})".format(i, score))

plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()



#44444444444444444444444444444444444444444444444444444444444444444444444
pca = PCA(2)
xPCA = pca.fit_transform(xTrain)
df2 = pd.DataFrame(data=xPCA)
finaldf = pd.concat([df2,data[['TENURE']]],axis=1)
print(finaldf)

wcss = []

#elbow method to figure out the number of clusters
for i in range(2,5):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(df2)

    wcss.append(kmeans.inertia_)
    score = silhouette_score(df2, kmeans.labels_, metric='euclidean')
    print("For n_clusters = {}, silhouette score = {})".format(i, score))
plt.plot(range(1, 4), wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()

#Bonus Bonus Bonus Bonus Bonus Bonus
pca = PCA(2)
xPCA = pca.fit_transform(xScaled)
df2 = pd.DataFrame(data=xPCA)
finaldf = pd.concat([df2,data[['TENURE']]],axis=1)
print(finaldf)

wcss = []

#elbow method to figure out the number of clusters
for i in range(2,5):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(df2)

    wcss.append(kmeans.inertia_)
    score = silhouette_score(df2, kmeans.labels_, metric='euclidean')
    print("For n_clusters = {}, silhouette score = {})".format(i, score))
plt.plot(range(1, 4), wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()
