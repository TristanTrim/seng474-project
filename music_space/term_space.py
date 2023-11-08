
import csv
import numpy as np
from sklearn.cluster import KMeans
"""
This file experiments with a music space containing the
terms (keyword) associated with a song
"""

def get_terms_feature_vectors():
    X = []
    with open('embeddings/csv/MSD_songs.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            X.append(row)

    return np.array(X)

    

X = np.array(get_terms_feature_vectors())