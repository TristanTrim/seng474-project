#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implement Exact Matrix Completion to estimate empty entries of score matrix in user_taste module
"""

import cvxpy as cp
import numpy as np
from user_taste.user_taste import user_taste

    

class MC_score_matrix(user_taste):
    
    def __init__(self,path):
        super().__init__(path)
        self.index_dictionary = self.__get_index_dictionary()
        self.__init_score_matrix(path)
        
        self.score_matrix = self.__MC_solve()
        
    def get_song_score(self,key):
        '''get song score from matrix'''
        
        index = self.index_dictionary(key)
        return self.score_matrix(index)

    def __init_score_matrix(self,path):
        '''initialize score matrix with given song scores from taste space'''
            
            
        # matrix will be indexed lexographically, rowwise by user, columwise by song
        user_count = len(self.get_all_users())
        song_count = len(self.get_all_songs())    
    
        self.score_matrix = np.zeros((user_count,song_count))
          
        for (key,index) in self.index_dictionary.items():
            self.__add_score(key, index)
        
        
        
        
    
    
    def __add_score(self,key,index):
        if key in self.taste_dictionary:
            self.score_matrix[index] = self.taste_dictionary[key]
            
            
        
    def __get_index_dictionary(self):
        '''
        create a dictionary that maps taste_dictionary (uid,sid) key values to score matrix indices (i,j)
        representing the rating of the jth song (sid) by the ith user (uid)
        '''
        users = sorted(self.get_all_users())
        songs = sorted(self.get_all_songs())
        
        indices = [(i,j) for i in range(len(users)) for j in range(len(songs))]
        keys = [(user,song) for user in users for song in songs]
        
        
        index_dictionary = dict(zip(keys,indices)) 
        return index_dictionary
    

    
    
    def __MC_solve(self):
        '''set-up and solve optimization problem and constraints for CVX'''
    
        X = cp.Variable(self.score_matrix.shape)
        objective = cp.Minimize(cp.atoms.normNuc(X))
        constraints = []
        
        # get nonzero matrix entries 
        I,J = np.where(self.score_matrix!=0)
        nonzero_entries = [(i,j) for (i,j) in zip(I,J)]
        
        # define constraints to optimization problem
        for (i,j) in nonzero_entries:
            C = X[i,j] == self.score_matrix[i,j]
            constraints.append(C)
        
        problem = cp.Problem(objective,constraints)
        
        print('Solving Matrix Completion problem:')
        problem.solve()
        
        return X.value
        
        
        
        