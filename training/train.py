import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import MultimodalSentimentModel
from dataset import MultimodalSentimentDataset
from utils import load_glove_embeddings, build_tokenizer
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import classification_report

# Hyperparameters
EMBED_DIM = 100
HIDDEN_DIM = 128
OUTPUT_DIM = 3
BATCH_SIZE = 16
EPOCHS = 5
MAX_LEN = 50
LEARNING_RATE = 1e-4

# Load GloVe and tokenizer
glove = load_glove_embeddings()
vocab = list(glove.keys())
tokenizer = build_tokenizer(vocab)
with open("../api/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Embedding matrix
vocab_size = len(tokenizer)
embedding_matrix = torch.zeros((vocab_size, EMBED_DIM))
for word, idx in tokenizer.items():
    if word in glove:
        embedding_matrix[idx] = torch.tensor(glove[word])

# Dataset & Dataloader
train_csv = "data/train.csv"
image_dir = "data/images"
dataset = MultimodalSentimentDataset(train_csv, image_dir, tokenizer, max_len=MAX_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultimodalSentimentModel(vocab_size, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, embedding_matrix).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for images, texts, labels in dataloader:
        images, texts, labels = images.to(device), texts.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")
    print(classification_report(all_labels, all_preds))

# Save model
torch.save(model.state_dict(), "../api/model.pt")
