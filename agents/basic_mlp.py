#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as tc

class Agent():

    def __init__(self):

        self.threshold = 0 ## TODO actual threshold based on the average score returned by get_song_score
        self._song_vec_len = 100 ## TODO actual len

        self._good_songs_vec = tc.zeros(
                        self._song_vec_len )
        self._bad_songs_vec = tc.zeros(
                        self._song_vec_len )

        self._mlp = # TODO neural net goes herererererer

    def get_next_recommendation(self, round_history):

        if ( round_history ):

            last_song = round_history[-1][0]
            last_score = round_history[-1][1]

            if ( last_score > self.threshold ):
                # that was a good song
                self._good_songs_vec += last_song
            else:
                # not a good song
                self._bad_songs_vec += last_song
        ##
        
        next_song = self._mlp.predict( tc.concat(
                    (
                        self._good_songs_vec,
                        self._bad_songs_vec,
                    )
                ))

        return( next_song )

        
