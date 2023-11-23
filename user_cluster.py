from user_taste.user_taste import user_taste
from music_space.music_space import music_space
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

UT = user_taste('user_taste/data/')
MS = music_space('music_space/embeddings/npy/')

import numpy as np

def uid_to_song_space(uid, MS, UT):
    return np.array([MS.songID_to_vector(sid) for sid in UT.get_listening_history(uid)[:, 1]])

def avg_song(D):
    return np.mean(D, axis=0)
#return a vector with each song score for that user


def score(uid, UT, MS,song_dict):
    uid_avg = avg_song(uid_to_song_space(uid,MS,UT))
    S = [1 / (np.linalg.norm(MS.songID_to_vector(song) - uid_avg)) for song in MS.song_IDs]
    S = np.array(S)
    for record in UT.get_listening_history(uid):
        S[song_dict[record[1]]] = record[2]

    return S

def create_score_matrix():
    UT = user_taste('user_taste/data/')
    MS = music_space('music_space/embeddings/npy/')
    S = np.zeros((UT.get_all_users().shape[0],MS.feature_vectors.shape[0]))
    song_dict = {song: idx for idx, song in enumerate(MS.song_IDs)}

    for idx, uid in enumerate(UT.get_all_users()):
        S[idx, :]= score(UT.get_rand_user(), UT,MS,song_dict)
        print(idx)
    

    np.save('music_space/embeddings/npy/score_matrix.npy', S)

X = np.load('music_space/embeddings/npy/score_matrix.npy')
X = np.reshape(X, X.shape[0]*X.shape[1])
print(X.shape)

import matplotlib.pyplot as plt
plt.hist(X,bins=20,range=(0,20))
plt.yscale('log')
plt.title("Distrubution of score matrix")


plt.show()