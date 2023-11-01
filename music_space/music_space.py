import numpy as np
from sklearn.neighbors import KDTree
import time
import matplotlib.pyplot as plt

class MusicSpace:

    def __init__(self, title):
        self.feature_vectors = np.load('embeddings/' + title + '_music_space.npy')
        self.song_IDs = np.load('embeddings/' + title + '_song_ID_space.npy', allow_pickle=True)
        self.KD_TREE = KDTree(data= self.feature_vectors, leaf_size= 1) #tune leaf size??

    def vector_to_songID(self, song_vector):
        """returns the song_ID for the given song_vector"""

        distances = np.linalg.norm(self.feature_vectors - song_vector, axis=1)
        min_distance_index = np.argmin(distances)        
        song_id = self.song_IDs[min_distance_index]
        
        return song_id

    def songID_to_vector(self, song_id):
        """returns the song_vector for the given song_id"""

        index = np.where(self.song_IDs == song_id)[0]
        if index.size == 0:
            return None
        else:
            vector = self.feature_vectors[index[0]]
            return vector

    def NN(self, song_vector):
        """returns the nearest neighbour of a song vector"""

        dist, index = self.KD_TREE.query(X = song_vector.reshape(1,-1), k = 2)
        print(f"distance = {dist[0][1]}")
        return(self.feature_vectors[index[0][1]])

#Example Usage

#initialize music space
t0 = time.time()
title = 'v1' #using the first music space version
MS = MusicSpace(title) #initialize the space
print(f"initialization time: {time.time() - t0}")


#Try a nearest Neighbour query on the 4th song in the music space
X1 = MS.feature_vectors[3]

t0 = time.time()
MS.NN(X1)
print(f"kNN query time: {time.time() - t0}")

