#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of env
"""
import numpy as np
import gym
from gym import spaces

from user_taste.user_taste import user_taste, MC_score_matrix
from music_space.music_space import music_space
from game_engine.game_engine import GameEngine
import torch

class DJEnv():
    def __init__(
            self,
            tastes_set=None,
            music_space=None,
            score_matrix = None,
            # possible env types:
            #   sum_of_songvec, n_back, last_score
            env_type = "sum_of_songvec",
            # possible stop types:
            #   num_steps, score
            stop_type="num_steps",
            stop_condition=30,
            song_vec_len = 40, ## TODO, actual len
            ):

        # properties
        
        self.tastes_set = None
        self.music_space = None
        self.score_matrix = None
    
        self._round_history = []
        self._uid = None

        self._env_type = env_type
        self._stop_type = stop_type
        self._stop_condition = stop_condition
        self._song_vec_len = song_vec_len

        self._num_steps = 0

        # set modules from input or default

        self.set_tastes_set(tastes_set)
        self.set_music_space(music_space)
        self.set_score_matrix(score_matrix)

        self._song_vec_len = self.music_space.songID_to_vector(self.music_space.get_random_song()).shape[0]# ah ha ha ah >:^D

        # env_type

        if (self._env_type == "sum_of_songvec"):
            # TODO figure out a reasonable value for
            # or way of calculating _threshold
            self._threshold = 0.9
            self._zero_good_bad_vec()
            action_shape = self._bad_songs_vec.shape
            observation_shape = self.get_sum_of_songvec().flatten().shape

        # gym properties

        self.action_space = spaces.Box(
                    low=0, high=1,  # action 1 for song picked zero for all else ?
                    shape=action_shape, dtype=np.float32)

        self.observation_space = spaces.Box(
                        low=-1, high=1, # -1 for not liked and 1 for liked ?
                        shape=observation_shape, dtype=np.float32)

        self.reward_range = (0, 35)


    # getters and setters
        
    def set_tastes_set(self, tastes_set):
        if tastes_set is None:
            tastes_set = user_taste("./user_taste/data/")
        self.tastes_set = tastes_set

    def set_music_space(self, local_var_music_space):
        if local_var_music_space is None:
            local_var_music_space = music_space("./music_space/embeddings/npy/")
        self.music_space = local_var_music_space

    def set_score_matrix(self, score_matrix):
        if score_matrix is None:
            score_matrix = MC_score_matrix('./user_taste/data/')
        self.score_matrix = score_matrix

    # real deal potatoes and meat

    def _zero_good_bad_vec(self):
        self._good_songs_vec = torch.zeros(
                        self._song_vec_len )
        self._bad_songs_vec = torch.zeros(
                        self._song_vec_len )

    def _update_good_bad(self, song, score ):
            if ( score > self._threshold ):
                # that was a good song
                self._good_songs_vec += song
            else:
                # not a good song
                self._bad_songs_vec += song


    def get_sum_of_songvec(self):
        return( np.array((
                    self._good_songs_vec,
                    self._bad_songs_vec,
                    ))
                )


    def reset(self):
        self.round_history = []
        if (self._env_type == "sum_of_songvec"):
            self._zero_good_bad_vec()
        self._num_steps = 0

        self._uid = self.tastes_set.get_rand_user()

        if (self._env_type == "sum_of_songvec"):
            obs = self.get_sum_of_songvec().flatten()

        return(obs)


    def step(self, action):

        self._num_steps += 1

        # get the score for the recommended song
        song_rec_vec = action
        sid = self.music_space.vector_to_songID(song_rec_vec)
        song_score = self.score_matrix.get_song_score(
                     (self._uid, sid) )

        # add new song and score to round history
        self.round_history += [(song_rec_vec, song_score)]


        # calculate the observation, reward, and finished state

        if (self._env_type == "sum_of_songvec"):
            self._update_good_bad(song_rec_vec, song_score)
            obs = self.get_sum_of_songvec().flatten()
        
        reward = song_score

        if self._stop_type == "num_steps":
            done = (self._num_steps == self._stop_condition)

        # TODO, find out what "_" is supposed to be
        _ = None

        return( obs, reward, done, _ )

