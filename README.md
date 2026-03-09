---
title: transformer-edu-viz
colorFrom: indigo
colorTo: purple
sdk: docker
---

<div align="center">
<h1>🤖 Transformer From Scratch — Educational Visualizer</h1>
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=6366F1&center=true&vCenter=true&width=700&lines=Train+a+Transformer+Step+by+Step;Live+Loss+Curves+%26+Attention+Heatmaps;English+%E2%86%92+French+Neural+Translation" alt="Typing SVG"/>

[![Python](https://img.shields.io/badge/Python-3.10+-3b82f6?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-4f46e5?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-3b82f6?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/mnoorchenar/spaces)
[![Status](https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge)](#)

**🤖 Transformer From Scratch** — An interactive, step-by-step educational visualizer that trains a Transformer model from scratch for English→French translation, with live loss curves, real-time sample translations, and attention heatmaps — all inside your browser.
</div>

## ✨ Features

| | Feature | Description |
|---|---|---|
| 📊 | **6-Step Wizard** | Guided pipeline: Data → Vocab → Model → Train → Evaluate → Infer |
| 📡 | **Live Training Stream** | Server-Sent Events stream real-time loss, LR, and translations per epoch |
| 🧠 | **Attention Heatmap** | Cross-attention weights visualized as an interactive color grid |
| 📈 | **Live Loss Chart** | Chart.js line chart updating in real-time with train and val loss |
| ⚙️ | **Configurable Model** | Adjust d_model, heads, layers, d_ff, dropout with live param count |
| 🔤 | **Greedy vs Beam** | Side-by-side comparison of decoding strategies with BLEU scores |

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────┐
│              transformer-edu-viz                        │
│                                                         │
│  ┌───────────┐    ┌───────────┐    ┌───────────────┐   │
│  │  80 EN→FR │───▶│ Transformer│───▶│  Flask + SSE  │   │
│  │   Pairs   │    │  (PyTorch) │    │   Backend     │   │
│  └───────────┘    └───────────┘    └───────┬───────┘   │
│                                            │            │
│                                   ┌────────▼────────┐   │
│                                   │  SPA Dashboard  │   │
│                                   │  Chart.js + JS  │   │
│                                   └─────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Getting Started
```bash
git clone https://github.com/mnoorchenar/transformer-edu-viz.git
cd transformer-edu-viz
python -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python app.py
```

Open `http://localhost:7860` 🎉

## 🐳 Docker Deployment
```bash
docker compose up --build
```

## 📊 Dashboard Modules

| Module | Description | Status |
|--------|-------------|--------|
| 📊 Data Explorer | Pair browser, augmentation config, dataset stats | ✅ Live |
| 📚 Vocabulary | Vocab builder with interactive tokenizer demo | ✅ Live |
| 🏗️ Model Config | Hyperparameter sliders with live parameter count | ✅ Live |
| 🎯 Live Training | SSE-driven loss chart, LR curve, live translations | ✅ Live |
| 📈 Evaluation | BLEU scores (greedy vs beam), translation table | ✅ Live |
| 🔤 Inference | Translate any sentence + cross-attention heatmap | ✅ Live |

## 🧠 ML Models
```python
models = {
    "architecture": "Vanilla Transformer (Vaswani et al. 2017)",
    "task": "Sequence-to-sequence Machine Translation",
    "decoding": ["Greedy Search", "Beam Search (beam=4)"],
    "loss": "Label Smoothing Cross-Entropy (KL Divergence)",
    "optimizer": "Adam with Warmup LR Schedule",
    "metric": "SacreBLEU corpus score"
}
```

## 📁 Project Structure
```
transformer-edu-viz/
├── 📂 transformer/
│   ├── 📄 __init__.py
│   ├── 📄 data.py          # Dataset, vocab, encode/decode
│   └── 📄 model.py         # Transformer, loss, scheduler
├── 📂 templates/
│   └── 📄 index.html       # Full SPA with Chart.js & canvas heatmap
├── 📄 app.py               # Flask app, SSE stream, decode helpers
├── 📄 Dockerfile
├── 📄 requirements.txt
└── 📄 README.md
```

## Disclaimer

This project is developed strictly for educational and research purposes. The dataset is hardcoded and synthetic. No real user data is stored. Provided "as is" without warranty of any kind.

## 📜 License

MIT License.