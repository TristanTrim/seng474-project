#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of user taste module

NOTE: check that entp.txt and uid.txt exist and are in appropriate directories before use. (uid.txt can be generated with entp_reader module)
"""
from numpy.random import randint

def get_random_user():
    '''select random user ID from ENTP (entp.txt and uid.txt must exist in data directory)'''
    
    UID_LENGTH = 1019318
    
    # select random user from uid.txt
    i = randint(0,UID_LENGTH-1)
    COUNTER = 0
    
    with open('./data/uid.txt') as file:
        for line in file:
            uid = line.strip()
            
            if i == COUNTER:
                random_user = uid
                return random_user
            
            COUNTER +=1
    
def get_song_scores(user_id):
    '''get all song id's and scores for given user'''
    
    song_score = []
    
    with open('./data/train_triplets.txt') as file:
        
        for line in file:
            uid,sid,score = line.strip().split()
            
            if uid == user_id:
                song_score.append((sid,int(score)))
            
    return song_score
                
            
    
class user_taste:
    '''user_taste stores the ENTP song score pairs for a particular user, each orrespondding number of times a user has listened to a particular song.
    
    '''
    
    def __init__(self,user_id = get_random_user()):
        '''
        Initialize user taste instance.
        
        Attributes:
         - user_id (str): unique identifier of the user, selected randomly by default
         - song_scores (list): list of tuples: containing MSD song ID (str) and corresponding number of listens by that user (int)
            
        '''
        
        self.user_id = user_id
        self.song_scores = get_song_scores(self.user_id)
        
    def __str__(self):
        string = 'User: ' + self.user_id + '\n'
        string += 'sid' + ' '*16 + 'score\n'
        string += '-'*25+'\n'
    
        for (sid,score) in self.song_scores:
            string += str(sid) + ' ' + str(score) + '\n'
            
        return string
        
    def get_score(self,sid):
        '''
        Get score of a song for a user given it's ID
        
        Parameters:
         - sid: song ID of interest
        
        Returns:
         - score (int): number of listens of song by user. Returns 0 if sid not found in song_score.
        '''
        
        for song_id,score in self.song_scores:
            if sid == song_id:
                return score
            
        return 0
        