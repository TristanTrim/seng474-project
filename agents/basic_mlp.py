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


    def fit(self, x, y, epochs = 10, val_x=None, val_y=None):

      self.init_weights()

      optimizer = torch.optim.SGD(self.parameters(), lr=self.alpha)
      dataloader = get_dataloader(x,y)
      loss_fn = tc.nn.CrossEntropyLoss()

      losses = np.array([])
      val_losses = np.array([])
      for t in range(epochs):

        if (val_x is not None and val_y is not None):
          train_loop(dataloader, self, loss_fn, optimizer)
          losses = np.append(losses, zoloss(self,x,y))
          val_losses = np.append(val_losses, zoloss(self,val_x,val_y) )
        else:
          losses = np.append(losses, train_loop(dataloader, self, loss_fn, optimizer).detach().numpy().squeeze() )

      if val_x is not None:
        return( losses, val_losses )
      return(losses)

   # def predict(self, X):
   #   output = self.forward(torch.Tensor(X))

   #   args = torch.argmax(logits,1)
   #   args*=2
   #   args+=5

   #   return( args.detach().numpy() )


# agent which wraps mlp. This is what you import and call!

class Agent():

    def __init__(self):

        self.threshold = 0 ## TODO actual threshold based on the average score returned by get_song_score
        self._song_vec_len = 40 ## TODO actual len

        self._good_songs_vec = tc.zeros(
                        self._song_vec_len )
        self._bad_songs_vec = tc.zeros(
                        self._song_vec_len )

        input_width = 2*self._song_vec_len
        output_width = self._song_vec_len
        
        # TODO hyperparammmmmers? Figure out from some formula???
        width = 256
        hiddenheight = 4

        self._mlp = NeuralNetwork( input_width, output_width, width, hiddenheight)

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
        
        next_song = self._mlp.forward( tc.concat(
                    (
                        self._good_songs_vec,
                        self._bad_songs_vec,
                    )
                ))

        return( next_song )

        
