from sklearn.cluster import KMeans
import numpy as np


def kMeans(x, k=3):
    kmeans = KMeans(n_clusters = k).fit(x)
    

def main():
    print("Hello, World!")

if __name__ == "__main__":
    main()
