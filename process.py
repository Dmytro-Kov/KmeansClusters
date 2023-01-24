import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt



def classify(point, centers):
    dist_fake = np.linalg.norm(point[:2]-centers[1])
    dist_real = np.linalg.norm(point[:2]-centers[0])
    label = 2 if dist_fake > dist_real else 1
    return label == point[2]






def main():
    data = pd.read_csv("labeled.csv")
    del data["V3"]
    del data["V4"]

    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)

    data["V1"] = (data["V1"] - np.min(data["V1"], 0)) / (np.max(data["V1"], 0) - np.min(data["V1"], 0))
    data["V2"] = (data["V2"] - np.min(data["V2"], 0)) / (np.max(data["V2"], 0) - np.min(data["V2"], 0))

    genuine = data[ data["Class"] == 1 ]
    forged = data[ data["Class"] == 2 ]

    stacked = np.column_stack((data["V1"], data["V2"]))
    km_res = KMeans(n_clusters=2).fit(stacked)
    centers = km_res.cluster_centers_

    fig, graph = plt.subplots()
    #graph.scatter(genuine["V1"], genuine["V2"], alpha=0.4)
    #graph.scatter(forged["V1"], forged["V2"], alpha=0.4)
    graph.scatter(centers[0,0], centers[0,1], s=100, color="orange")   #fake
    graph.scatter(centers[1,0], centers[1,1], s=100, color="blue") #true
    plt.xlabel("variance")
    plt.ylabel("skewness")

    count = 0
    for k,row in data.iterrows():
        if classify(row, centers):
            graph.scatter(row["V1"], row["V2"], alpha=0.4, color="green")
            count += 1
        else:
            graph.scatter(row["V1"], row["V2"], alpha=0.4, color="red")

    print(count/data.shape[0])

    plt.show()


main()