import numpy
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

churn = pd.read_csv("/Users/admin/Downloads/Churn_Modelling.csv")
churn = churn.iloc[:, 2:]
churn = churn[churn['Exited'] == 1]
churn = churn.iloc[:,1:11]

gender = {'Male': 1,'Female': 2}
geograph = {'France':1,'Spain':2,'Germany':3}
churn.Gender = [gender[item] for item in churn.Gender]
churn.Geography = [geograph[item] for item in churn.Geography]
print(churn)
# wcss = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
#     kmeans.fit(churn)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 11), wcss)
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()

# kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
# pred_y = kmeans.fit_predict(churn)
# plt.scatter(churn[:,0], churn[:,1])
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
# plt.show()

km= KMeans(n_clusters=2)
clusters=km.fit_predict(churn)
churn['clusters']=clusters
# plt.scatter(x=churn.iloc[:,:], y=churn[:,10], c=churn["clusters"], s=30)
# plt.show()
print(churn)
#import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
pca = PCA(2)

plot_columns = pca.fit_transform(churn)
print(plot_columns)

# Plot based on the two dimensions, and shade by cluster label
plt.scatter(x=plot_columns[:,1], y=plot_columns[:,0], c=churn["clusters"], s=30)
plt.show()

print(km.cluster_centers_)