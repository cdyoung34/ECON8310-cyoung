import torch
import torch.optim as optim
from MyNeuralNet import MyNeuralNet

# Import model
model = MyNeuralNet()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Load the checkpoint
checkpoint = torch.load('model.pt', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']