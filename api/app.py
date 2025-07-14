from flask import Flask, request, jsonify
import torch
from PIL import Image
from torchvision import transforms
import pickle
from io import BytesIO
from training.model import MultimodalSentimentModel
from training.utils import tokenize_text
import numpy as np

# Constants
EMBED_DIM = 100
HIDDEN_DIM = 128
OUTPUT_DIM = 3
MAX_LEN = 50

# Load tokenizer and embeddings
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

embedding_matrix = torch.zeros((len(tokenizer), EMBED_DIM))
# Optionally load precomputed GloVe (assume saved or recomputed elsewhere)

# Model setup
model = MultimodalSentimentModel(len(tokenizer), EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, embedding_matrix)
model.load_state_dict(torch.load("model.pt", map_location=torch.device("cpu")))
model.eval()

# Flask app
app = Flask(__name__)

# Image transform
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files or 'text' not in request.form:
        return jsonify({"error": "Missing image or text input"}), 400

    image_file = request.files['image']
    text = request.form['text']

    image = Image.open(BytesIO(image_file.read())).convert('RGB')
    image_tensor = image_transform(image).unsqueeze(0)

    tokens = tokenize_text(text, tokenizer, MAX_LEN)
    text_tensor = torch.tensor(tokens).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor, text_tensor)
        probs = torch.softmax(output, dim=1).squeeze()
        pred_class = torch.argmax(probs).item()

    label_map = {0: "negative", 1: "neutral", 2: "positive"}

    return jsonify({
        "prediction": label_map[pred_class],
        "probabilities": {
            "negative": round(probs[0].item(), 4),
            "neutral": round(probs[1].item(), 4),
            "positive": round(probs[2].item(), 4)
        }
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
