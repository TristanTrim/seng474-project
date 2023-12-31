#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as tc
tc.autograd.set_detect_anomaly(True)

device = (
    "cuda" if tc.cuda.is_available()
    else "mps" if tc.backends.mps.is_available()
    else "cpu"
)
print(f"Agent: Using {device} device")

# neural network for agent (down below)

class NeuralNetwork(tc.nn.Module):

    def __init__(self, input_width, output_width, width, hiddenheight):
        super().__init__()

        self.flatten = tc.nn.Flatten()

        # input layer
        layers = [tc.nn.Linear(input_width, width)]

        # hidden layers
        for i in range(hiddenheight):
          layers+=[tc.nn.ReLU(),
                   tc.nn.Linear(width,width)
                   ]
 
        # output layer
        layers += [tc.nn.ReLU(),
                   tc.nn.Linear(width,output_width)
                   ]

        self.linear_relu_stack = tc.nn.Sequential( *layers )

        self.init_weights()


    def forward(self, x):
        #x = self.flatten(x)
        output = self.linear_relu_stack(x)
        return output


    def init_weights(self):
      for layer in self.linear_relu_stack:
        if isinstance(layer, tc.nn.Linear):
          tc.nn.init.xavier_normal(layer.weight)

# agent which wraps mlp. This is what you import and call!

class Agent():

    def __init__(self):

        self.threshold = 0 ## TODO actual threshold based on the average score returned by get_song_score
        self._song_vec_len = 40 ## TODO actual len

        self._zero_good_bad_vec()

        input_width = 2*self._song_vec_len
        output_width = 2*self._song_vec_len
        
        # TODO hyperparammmmmers? Figure out from some formula???
        width = 256
        hiddenheight = 4

        self._mlp = NeuralNetwork( input_width, output_width, width, hiddenheight)

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
        ##

    def _get_mu_sig(self):

        _input = tc.detach( tc.concat(
                    ( self._good_songs_vec, self._bad_songs_vec, )
                ) )

 
        output = self._mlp.forward(_input)

        mu = output[:self._song_vec_len]
        sig = tc.sigmoid( output[self._song_vec_len:]*1e-6 )*1e3

        return( mu, sig )

    def get_next_recommendation(self, round_history):

        if ( round_history ):

            last_song = round_history[-1][0]
            last_score = round_history[-1][1]

            self._update_good_bad(last_song, last_score)


        mu, sig = self._get_mu_sig()

        next_song = tc.normal(mu,sig)

        return( next_song )

    def update_weights(self, round_returns, alpha ):

        self._zero_good_bad_vec()
        optimizer = tc.optim.SGD(
                self._mlp.parameters(), lr=alpha)

        for song, score, _return in round_returns:

            mu, sig = self._get_mu_sig()

                # Really high score gets close to 1.
                # Positive scores are move in direction
                # of predicting that songvec.
            gb = tc.sigmoid(tc.Tensor((_return,)) - self.threshold)[0]*2-1
            delta = song - mu
            better_mu = mu + gb * delta

            better_delta = song - better_mu
            better_sig = sig * (delta/better_delta)

            loss = tc.nn.functional.mse_loss(
                            tc.concat((mu,sig)),
                            tc.concat((better_mu,better_sig))
                            )
            print(f"score: {score}, loss: {loss}")

            loss.backward()
            optimizer.zero_grad()
            optimizer.step()
            
            self._update_good_bad(song, score)


