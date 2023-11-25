import numpy as np
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of env
"""

class DJEnv():
    def __init__(
            tastes_set=None,
            music_space=None,
            score_matrix = None,
            # possible env types:
            #   sum_of_songvec, n_back, last_score
            env_type = "sum_of_songvec",
            song_vec_len = = 40, ## TODO, actual len
            ):


        # properties
        
        self.agent = None
        self.tastes_set = None
        self.music_space = None
        self.score_matrix = None
    
        self.round_history = []
        self.uid = None
        self._

        # set modules from input or default

        self.set_agent(agent)
        self.set_tastes_set(tastes_set)
        self.set_music_space(music_space)
        self.set_score_matrix(score_matrix)
        

        # env_type
        if (env_type == "sum_of_songvec"):
            self._zero_good_bad_vec()

    # getters and setters
        
    def set_agent(self, agent):
        self.agent = agent

    def set_tastes_set(self, tastes_set):
        self.tastes_set = tastes_set

    def set_music_space(self, music_space):
        self.music_space = music_space

    def set_score_matrix(self, score_matrix):
        self.score_matrix = score_matrix

    # real deal potatoes and meat

    def _zero_good_bad_vec(self):
        self._good_songs_vec = tc.zeros(
                        self._song_vec_len )
        self._bad_songs_vec = tc.zeros(
                        self._song_vec_len )

    def _update_good_bad(self, song, score ):
            if ( score > self.threshold ):
                # that was a good song
                self._good_songs_vec += song
            else:
                # not a good song
                self._bad_songs_vec += song


    def get_sum_of_songvec(self):
        return( 

    def step(self, action):

    def reset(self):
        self.round_history = []
        if (env_type == "sum_of_songvec"):
            self._zero_good_bad_vec()

### TODO finish
        if ( self.round_history ):

            last_song = round_history[-1][0]
            last_score = round_history[-1][1]

            self._update_good_bad(last_song, last_score)
        
    # Attributes <- will need to move this up to the innit I think 

    self.action_space = np.arrange(614) # creates an array {0,1,...,614} for each song 

    self.observation_space = # Does this require a matrix of all users and songs and if they like the song or not?

    self.reward_range = (-2.755e-05, 35)