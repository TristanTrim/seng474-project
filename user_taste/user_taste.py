#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Implementation of user taste module

NOTE: This module requires the user taste dataset to be available. This file can be generated by running entp_reader.py


"""
import numpy as np 
from numpy.random import randint
import matplotlib.pyplot as plt

class user_taste():

    def __init__(self,path):
        self.taste_space = np.load(path + 'user_taste.npy')
        self.taste_dictionary = self.__init_taste_dictionary()


    def __init_taste_dictionary(self):
        """constructs a dictionary of (sid,uid) : score pairs"""
        #group the sid and uid together into tuples
        keys = map(tuple, self.taste_space[:, :2])

        #cast the scores to int
        values = self.taste_space[:, 2].astype(int)

        taste_dictionary = dict(zip(keys, values))
        return taste_dictionary

    def get_song_score(self,uid,sid):
        """return the score given a user id (uid) and song id (sid)"""

        listening_history = self.get_listening_history(uid)
        for song in list(listening_history[:, 1]):
            if song == sid:
                return self.taste_dictionary[(uid,sid)]

        return 0
    
    def get_rand_user(self):
        """returns the user id (uid) of a random user in the user taste dataframe"""
    
        i = randint(0,self.taste_space.shape[0]-1)
        return self.taste_space[i,0]
    
    def get_listening_history(self,uid):
        """return all the songs which a given user has listened to"""
        uid_records = self.taste_space[:,0] == uid
        return self.taste_space[uid_records]

    def get_all_users(self):
        return set(self.taste_space[:,0])
        



