#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np



def process(entp_file):
    """
        This function extracts the raw song id, user id, score triplets.
            - Only the triplets whose song id is found in the million song subset are kept
            - the resulting dataset is stored at user_taste/data/user_taste.npy

        
    """
    # retrieve unique user and song id's
    user_taste = []
    scores = []

    MSD_song_ids = np.load('../music_space/embeddings/npy/MSD_song_IDs_2.npy')
    MSD_song_ids = MSD_song_ids.astype(str)  # Convert MSD_song_ids to string type

    i = 1
    with open(entp_file, 'r') as input_file:

        for line in input_file:

            uid, sid, score = line.strip().split()
            user_taste.append([uid, sid, score])
            scores.append(score)

            print(f"processed = {i}")
            i+=1
            #this was implemented because creating the whole dataset will 2 hours or so...
            if i == 1000000:
                break
    
    user_taste = np.array(user_taste)
    user_taste = user_taste[np.isin(user_taste[:, 1], MSD_song_ids)]
    np.save(file = 'data/user_taste.npy',arr = user_taste)

if __name__=="__main__":
    process("data/train_triplets.txt")
