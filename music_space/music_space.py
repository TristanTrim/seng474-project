import hdf5_getters as h5
import matplotlib.pyplot as plt 
import numpy as np
import time
import os

def grab_song(path):    
    song_file = h5.open_h5_file_read(path)

    song_file.close()


def initialize_song_space():

    """
    returns a pandas dataframe where each row is the selected attributes of the song.
    This fuction depends of the Million Song Subset being accessible 

    """

    root = "MillionSongSubset"
    capital_letters = [chr(i) for i in range(65, 91)]
    for pref1 in ['A', 'B']:
        for pref2 in capital_letters:
            for pref3 in capital_letters:
                directory = root + '/' + pref1 + '/' + pref2 + '/' + pref3
                for file_suffix in os.listdir(directory):
                    path = directory + '/' + file_suffix
                    grab_song(path)
                
                print(f"{len(os.listdir(directory))} songs processed")


all_songs()