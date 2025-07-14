import nltk
import torch
import numpy as np
import pickle
nltk.download('punkt')

from nltk.tokenize import word_tokenize

EMBEDDING_DIM = 100


def load_glove_embeddings(glove_path='glove.6B.100d.txt'):
    embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.array(parts[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings


def build_tokenizer(vocab, save_path='tokenizer.pkl'):
    word2idx = {word: idx + 1 for idx, word in enumerate(vocab)}
    word2idx['<PAD>'] = 0
    with open(save_path, 'wb') as f:
        pickle.dump(word2idx, f)
    return word2idx


def tokenize_text(text, tokenizer, max_len=50):
    tokens = word_tokenize(text.lower())
    token_ids = [tokenizer.get(t, 0) for t in tokens[:max_len]]
    token_ids += [0] * (max_len - len(token_ids))
    return token_ids