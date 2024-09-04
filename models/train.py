# models/train.py
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.data_loader import Flickr8kDataset
from models.model import ImageCaptioningModel

# Hyperparameters
embed_size = 256
hidden_size = 512
vocab_size = 5000  # Adjust based on your actual vocabulary size
num_layers = 1
learning_rate = 0.001
batch_size = 32
num_epochs = 10

# Load Dataset
train_dataset = Flickr8kDataset(root_dir='data/Flickr8k/', vocab_file='data/vocabulary.pkl')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize Model, Loss Function, Optimizer
model = ImageCaptioningModel(embed_size, hidden_size, vocab_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    for i, (images, captions) in enumerate(train_loader):
        outputs = model(images, captions[:, :-1])
        loss = criterion(outputs.view(-1, vocab_size), captions[:, 1:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")
