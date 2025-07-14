import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import pandas as pd
from utils import tokenize_text

class MultimodalSentimentDataset(Dataset):
    def __init__(self, csv_path, image_dir, tokenizer, max_len=50):
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row['image_name'])
        image = Image.open(img_path).convert('RGB')
        image = self.image_transform(image)

        text = row['text']
        tokens = tokenize_text(text, self.tokenizer, self.max_len)
        label = int(row['label'])  # 0 = negative, 1 = neutral, 2 = positive

        return image, torch.tensor(tokens), torch.tensor(label)