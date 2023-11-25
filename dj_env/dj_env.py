
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
            # possible stop types:
            #   num_steps, score
            stop_type="num_steps",
            stop_condition=30,
            song_vec_len = = 40, ## TODO, actual len
            ):


        # properties
        
        self.agent = None
        self.tastes_set = None
        self.music_space = None
        self.score_matrix = None
    
        self._round_history = []
        self._uid = None
        self._env_type = env_type
        self._stop_type = stop_type
        self._stop_condition = stop_condition
        self._num_steps = 0

        # set modules from input or default

        self.set_agent(agent)
        self.set_tastes_set(tastes_set)
        self.set_music_space(music_space)
        self.set_score_matrix(score_matrix)
        

        # env_type
        if (env_type == "sum_of_songvec"):
            # TODO figure out a reasonable value for
            # or way of calculating _threshold
            self._threshold = 0.9
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
        if (env_type == "sum_of_songvec"):
            self._zero_good_bad_vec()
        self._num_steps = 0

    def step(self, action):

        self._num_steps += 1

        # get the score for the recommended song
        song_rec_vec = action
        sid = self.music_space.vector_to_songID(song_rec_vec.detach().numpy())
        song_score = self.score_matrix.get_song_score(
                     (self._uid, sid) )

        # add new song and score to round history
        self.round_history += [(song_rec_vec, song_score)]


        # calculate the observation, reward, and finished state

        if (env_type == "sum_of_songvec"):
            self._update_good_bad(song_rec_vec, song_score)
            obs = self.get_sum_of_songvec()
        
        reward = song_score

        if self._stop_type == "num_steps":
            done = (self._num_steps == self._stop_condition)

        # TODO, find out what "_" is supposed to be
        _ = None

        return( obs, reward, done, _ )


