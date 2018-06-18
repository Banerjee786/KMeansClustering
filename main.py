

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd

df = pd.read_excel('titanic.xls')
#print(df.head(10))
df.drop(['body','name'], 1, inplace=True)
df.infer_objects()
df.fillna(0, inplace=True)
#print(df.head(10))

def handle_non_numeric_data(df):
    columns = df.columns.values
    for col in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]
        if df[col].dtype != np.int64 and df[col].dtype != np.float64:
            column_contents = df[col].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1
            df[col] = list(map(convert_to_int, df[col]))
    return df

df = handle_non_numeric_data(df)
#print(df.head(10))

df.drop(['boat'], 1, inplace=True)
#print(df.head(10))
#X = np.array(df.drop(['survived'], 1).astype(float))
X = np.array(df.drop(['sex'], 1).astype(float))
#X = np.array(df)
#X = preprocessing.scale(X)
Y = np.array(df['survived'])

clf = KMeans(n_clusters=10)
clf.fit(X)
centroids = clf.cluster_centers_
labels = clf.labels_
#colors = ["g.","b.","r.","c.","y."]
for i in range(len(X)):
    #print("Coordinate : ", X[i]," Label : ", labels[i])
    #plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 5)
    plt.plot(X[i][0], X[i][1], markersize=5)
    plt.ylabel('Survivors')
plt.scatter(centroids[:, 0], centroids[:, 1], marker=".", s=50, linewidths=2, zorder=2)
plt.show()
'''
correct = 1
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == Y[i]:
        correct = correct + 1

print "Prediction Result : ",(correct/len(X))
'''


