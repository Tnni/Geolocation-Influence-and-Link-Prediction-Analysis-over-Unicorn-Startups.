from time import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def isInt(value):
  try:
    int(value)
    return True
  except ValueError:
    return False

file1 = open('embeddings', 'r')
lines = file1.readlines()
embeddings = []
for line in lines:
    embedding = line.split(' ')
    embeddings.append(embedding)

del embeddings[0]
final_embeddings = {}
for e in embeddings:
    i = 0
    name = []
    while not isfloat(e[i]) or isInt(e[i]):
        name.append(e[i])
        i += 1
    name = ' '.join(name)
    temp = []
    while i < len(e):
        temp.append(float(e[i].strip('\n')))
        i += 1
    final_embeddings[name] = temp

X = []
y = []
for key in final_embeddings:
    X.append(final_embeddings[key])
    y.append(key)

X = np.array(X)
X = X / X.max(axis=0)
data = scale(X)

#
# n_samples, n_features = data.shape
# n_digits = len(y)
# labels = y
#
# sample_size = n_digits
#
# print("n_digits: %d, \t n_samples %d, \t n_features %d"
#       % (n_digits, n_samples, n_features))
#
#
# print(82 * '_')
# print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')
#
#
# def bench_k_means(estimator, name, data):
#     t0 = time()
#     estimator.fit(data)
#     print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
#           % (name, (time() - t0), estimator.inertia_,
#              metrics.homogeneity_score(labels, estimator.labels_),
#              metrics.completeness_score(labels, estimator.labels_),
#              metrics.v_measure_score(labels, estimator.labels_),
#              metrics.adjusted_rand_score(labels, estimator.labels_),
#              metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
#              metrics.silhouette_score(data, estimator.labels_,
#                                       metric='euclidean',
#                                       sample_size=sample_size)))
#
# bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
#               name="k-means++", data=data)
#
# bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
#               name="random", data=data)
#
# # in this case the seeding of the centers is deterministic, hence we run the
# # kmeans algorithm only once with n_init=1
# pca = PCA(n_components=n_digits).fit(data)
# bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
#               name="PCA-based",
#               data=data)
# print(82 * '_')
#
# # #############################################################################
# # Visualize the results on PCA-reduced data
#
# reduced_data = PCA(n_components=2).fit_transform(data)
# kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
# kmeans.fit(reduced_data)
#
# # Step size of the mesh. Decrease to increase the quality of the VQ.
# h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
#
# # Plot the decision boundary. For that, we will assign a color to each
# x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
# y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#
# # Obtain labels for each point in mesh. Use last trained model.
# Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
#
# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.figure(1)
# plt.clf()
# plt.imshow(Z, interpolation='nearest',
#            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
#            cmap=plt.cm.Paired,
#            aspect='auto', origin='lower')
#
# plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# # Plot the centroids as a white X
# centroids = kmeans.cluster_centers_
# plt.scatter(centroids[:, 0], centroids[:, 1],
#             marker='x', s=169, linewidths=3,
#             color='w', zorder=10)
# plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
#           'Centroids are marked with white cross')
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())
# plt.show()

km = KMeans(
    n_clusters=10, init='random',
    n_init=10, max_iter=300,
    tol=1e-04, random_state=0
)
y_km = km.fit_predict(X)

pca = PCA(n_components=2)
a = pca.fit_transform(X)
x_vals, y_vals = a.T
s = [1 for _ in range(len(x_vals))]

colors = {}
for i, name in enumerate(y):
    colors[name] = y_km[i]
with open('cluster_coloring.pkl', 'wb') as f:
    pickle.dump(colors, f, pickle.HIGHEST_PROTOCOL)
plt.scatter(x_vals, y_vals, s=s, c=y_km)
plt.show()