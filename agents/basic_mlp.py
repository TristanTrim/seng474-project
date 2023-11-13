#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as tc

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

    def _update_good_bad( song, score ):
            if ( score > self.threshold ):
                # that was a good song
                self._good_songs_vec += song
            else:
                # not a good song
                self._bad_songs_vec += song
        ##

    def _get_mu_sig(self):

        output = self._mlp.forward( tc.concat(
                    (
                        self._good_songs_vec,
                        self._bad_songs_vec,
                    )
                ))
        mu = output[:self._song_vec_len]
        sig = output[self._song_vec_len:]

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
        optimizer = torch.optim.SGD(self._mlp.parameters(), lr=alpha)

        for song, score, _return in round_returns:

            mu, sig = self._get_mu_sig()

            goodness = _return - self.threshold
            delta = song - mu
            better_mu = mu + goodness * delta

            better_delta = song - better_mu
            better_sig = sig * (delta/better_delta)

            loss = tc.nn.functional.mse_loss(
                            tc.concat(mu,sig),
                            tc.concat(better_mu,better_sig))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            self._update_good_bad(song, score)


