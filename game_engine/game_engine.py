
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of user game engine module
"""

from user_taste.user import USER
class GameEngine():

    # game engine initialization

    def __init__(self,
            agent=None,
            tastes_set=None,
            music_space=None,
            score_matrix = None,
            testmode=False
            ):

        # properties
        
        self.agent = None
        self.tastes_set = None
        self.music_space = None
        self.score_matrix = None
        self.uid = None

        self.testmode = testmode
        if testmode: return

        # set modules from input or default

        if agent:
            self.set_agent(agent)
        else:
            raise Exception("No agent given and no default agent")

        if tastes_set:
            self.set_tastes_set(tastes_set)
        else:
            raise Exception("No tastes_set given and no default")
        

        if music_space:
            self.set_music_space(music_space)
        else:
            raise Exception("No music_space given and no default")
        
        if score_matrix:
            self.set_score_matrix(score_matrix)
        else:
            raise Exception("No score matrix given and no default")

    # getters and setters
        
    def set_agent(self, agent):
        self.agent = agent

    def set_tastes_set(self, tastes_set):
        self.tastes_set = tastes_set

    def set_music_space(self, music_space):
        self.music_space = music_space

    def set_score_matrix(self, score_matrix):
        self.score_matrix = score_matrix

    # meat and potatoes
    # ( code for actually doing stuff ) 

    def run_game(self,
                stop_condition=30, stop_mode="num_game_rounds",
                new_user=True,
                ):

        if self.testmode:
            return([("sandstorm", 9001), ("never gonna give you up", -777)])

        if new_user or not self.uid:
            self.uid = self.tastes_set.get_rand_user()
        print(f"running game with user {self.uid} (don't tell the agent)")

        round_history = []

        if stop_mode == "num_game_rounds":

            assert type(stop_condition) is int
            
            for round_number in range(1,stop_condition+1):
                round_history += [
                        self._run_game_round(round_history, self.uid) ]

            return( round_history )

        else:
            raise Exception(f"Unrecognised stop_mode: {stop_mode}")

    def _run_game_round(self, round_history, uid):

        song_rec_vec = self.agent.get_next_recommendation( round_history )
        sid = self.music_space.vector_to_songID(song_rec_vec.detach().numpy())
        song_score = self.score_matrix.get_song_score( (uid, sid) )
        return (song_rec_vec, song_score)

