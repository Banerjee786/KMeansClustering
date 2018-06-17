
from flask import Flask, render_template
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd

app = Flask(__name__)

@app.route('/')
def display():
    X = np.array([[2, 2],
                  [3, 3],
                  [2, 1],
                  [1, 2],
                  [10, 12],
                  [12, 11],
                  [9, 10],
                  [14, 15],
                  [14, 16],
                  [15, 14]
                  ])
    num_clusters = 3
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    inertia = kmeans.inertia_
    colors = ["g.", "b.", "m.", "y.", "c."]
    print("Centroids : ", centroids)
    print("Labels : ", labels)
    for i in range(len(X)):
        # print "Coordinate {} Label : {} is {}".format(i, labels[i], X[i])
        plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)
    base_centroid_x = centroids[0][0]
    base_centroid_y = centroids[0][1]
    icdx = []
    icdy = []
    for j in range(len(centroids)):
        inter_centroid_dist_x = abs(centroids[j][0] - base_centroid_x)
        icdx.append(inter_centroid_dist_x)
        inter_centroid_dist_y = abs(centroids[j][1] - base_centroid_y)
        icdy.append(inter_centroid_dist_y)
        print("Intercentroid Dist : {}, {}".format(inter_centroid_dist_x, inter_centroid_dist_y))
    print("Inertia : ", inertia)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=50, linewidths=5, zorder=5)
    plt.savefig('ClusteringFigure1.png', bbox_inches='tight')
    # plt.show()
    return render_template('firstpage.html',icdx=icdx,icdy=icdy,centroids=centroids,labels=labels,ln=len(centroids))

if __name__ == '__main__':
    app.run(debug=True)




