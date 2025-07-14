# 🤖 Multimodal Sentiment Analyzer

**Multimodal Sentiment Analyzer** is a production-ready, enterprise-grade deep learning project that performs sentiment analysis by jointly analyzing textual comments and accompanying images.

This project demonstrates the fusion of visual and linguistic modalities using PyTorch and advanced model design, exposed via a web interface and monitored using Prometheus and Grafana.

---

## 🔧 Features

- 🔀 **Multimodal fusion** (ResNet + GloVe + GRU + Gated Fusion)
- 📊 **Sentiment classification** (positive / neutral / negative)
- 🧠 **PyTorch model** with pretrained components
- 🌍 **Web interface** for uploading image and comment
- 📡 **Flask API** for prediction serving
- 📈 **Monitoring** with Prometheus + Grafana dashboards
- 🐳 **Containerized deployment** with Docker Compose

---

## 🚀 Quickstart

### 1. Clone & Setup
```bash
git clone https://github.com/your-username/multimodal-sentiment-analyzer.git
cd multimodal-sentiment-analyzer
```

### 2. Train the Model
```bash
cd training
python train.py
```

### 3. Launch the System
```bash
docker-compose up --build
```

### 4. Access Interfaces
- Webapp: [http://localhost:5000](http://localhost:5000)
- Prometheus: [http://localhost:9090](http://localhost:9090)
- Grafana: [http://localhost:3000](http://localhost:3000)
  - User: `admin`, Password: `admin`

---

## 📊 Example
Upload a tweet and image like:
- Text: *"Worst hotel breakfast ever. Cold coffee and dirty tables."*
- Image: blurry food tray

Output:
```json
{
  "prediction": "negative",
  "probabilities": {
    "negative": 0.89,
    "neutral": 0.07,
    "positive": 0.04
  }
}
```

---

## 📁 Architecture Overview

```text
📦 Multimodal Sentiment Analyzer
 ┣ 📂 training
 ┃ ┗━ PyTorch model + dataset + tokenizer
 ┣ 📂 api
 ┃ ┗━ Flask prediction endpoint
 ┣ 📂 webapp
 ┃ ┗━ Upload form + Chart.js + Bootstrap
 ┣ 📂 monitoring
 ┃ ┗━ Prometheus + Grafana setup
 ┗ 📄 docker-compose.yml
```

---

## 🎓 Motivation
This project shows expertise in:
- Deep learning with PyTorch
- Multimodal modeling and embeddings
- ML system deployment and monitoring
- Full-stack ML applications

Ideal for showcasing in a professional portfolio or technical interview.
