
import torch
import torch.nn as nn
import torchvision.models as models

class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = models.vgg16(pretrained=True).features  # Using VGG16 for CNN feature extraction
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(512 * 7 * 7, embed_size)  # Fully connected layer to transform features
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, images, captions):
        # Extract features using CNN
        with torch.no_grad():
            features = self.encoder(images)
        features = self.avgpool(features)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        features = self.relu(features)

        # Generate sequences using LSTM
        embeddings = torch.cat((features.unsqueeze(1), captions), 1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)

        return outputs

# Usage example:
# model = ImageCaptioningModel(embed_size=256, hidden_size=512, vocab_size=len(vocab))
