# utils/data_loader.py
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pickle

class Flickr8kDataset(Dataset):
    def __init__(self, root_dir, vocab_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        with open(vocab_file, 'rb') as f:
            self.vocab = pickle.load(f)
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        self.captions = self.load_captions()

    def load_captions(self):
        # Load and preprocess captions, converting to numerical representations
        captions = {}
        # Sample code to read captions, implement according to your dataset format
        with open(os.path.join(self.root_dir, 'captions.txt'), 'r') as f:
            for line in f:
                image, caption = line.strip().split('\t')
                if image in self.image_files:
                    captions[image] = self.vocab.encode(caption)
        return captions

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.root_dir, image_file)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        caption = self.captions[image_file]
        return image, torch.tensor(caption)
