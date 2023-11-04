import hdf5_getters as h5
import matplotlib.pyplot as plt 
import numpy as np
import os
from sklearn.neighbors import KDTree
import time 


def grab_song(path, row_ID):    

    """
    Extract the features from a song file (.h5 format) given a file path
    file path must be relative to music space

    returns:
        X: vector of song features
        song_ID: unique string identifier for the song
    """
    song_file = h5.open_h5_file_read(path)

    song_ID = h5.get_song_id(song_file)
    X = np.array(h5.get_segments_start(song_file)[:100])
    # X = np.array(
    #     [
    #         h5.get_danceability(song_file),
    #         h5.get_energy(song_file),
    #         h5.get_key(song_file),
    #         h5.get_mode(song_file),
    #         h5.get_tempo(song_file) 
    #     ]
    #     )
        
    # X = np.concatenate(
    #     (
    #         X,
    #         np.array(h5.get_segments_start(song_file)[:100])
    #     ),
    #     axis=0
    # )

    song_file.close()
    return [song_ID, X]


def num_features():
    return grab_song("MillionSongSubset\B\D\A\TRBDAID128F92E88C6.h5", 0)[1].shape[0]
    

def initialize_music_space():

    """
    1. Initialize and save the music space to music_space/embedding.
    Embedding is a numpy ndarray, saved as a .npy file
    The features in the music space are selected in the function grab song

    2. Initialize and save a vector of song_ID
    """

    root = "MillionSongSubset"
    processed = 0
    # Initialize an empty numpy ndarray with 10000 rows and n columns. 
    # Here, assuming a song has 'n' features.
    music_space = []
    song_IDs = []
    d = num_features()
    print(f"songs processed: {processed}",end='\r')

    for prefix_1 in ['A']:
        child = root + '/' + prefix_1

        for prefix_2 in os.listdir(child):
            g_child = child + '/' + prefix_2
            
            for prefix_3 in os.listdir(g_child):
                gg_child = g_child + '/' + prefix_3
            
                for song_file_path in os.listdir(gg_child):
        
                    song_ID, X = grab_song(path= gg_child + '/' + song_file_path, row_ID=processed)
                    
                    #convert the songID from numpy.bytes to string
                    song_ID = song_ID.decode("UTF-8")
                    if len(X) == d:
                        music_space.append(X)
                        song_IDs.append(song_ID)
                    processed += 1

                    print(f"songs processed: {processed}",end='\r')
                


    song_IDs = np.array(song_IDs)
    music_space = np.array(music_space)

    np.save(
        file = os.getcwd() + '/embeddings/MSD_features',
        arr = music_space
        )

    np.save(
        file = os.getcwd() + '/embeddings/MSD_song_IDs',
        arr = song_IDs
        )


initialize_music_space()
