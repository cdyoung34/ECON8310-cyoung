import torch
import torch.nn as nn


class MyNeuralNet(nn.Module):
    def __init__(self):
      # We define the components of our model here
      super(MyNeuralNet, self).__init__()
      # Function to flatten our image
      self.flatten = nn.Flatten()
      # Create the sequence of our network
      self.linear_relu_model = nn.Sequential(
            # Add a linear output layer w/ 10 perceptrons
            nn.LazyLinear(10),
        )
      
    def forward(self, x):
      # We construct the sequencing of our model here
      x = self.flatten(x)
      # Pass flattened images through our sequence
      output = self.linear_relu_model(x)

      # Return the evaluations of our ten 
      #   classes as a 10-dimensional vector
      return output

# Create an instance of our model
model = MyNeuralNet()