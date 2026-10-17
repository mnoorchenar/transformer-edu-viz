---
title: transformer-edu-viz
colorFrom: indigo
colorTo: purple
sdk: docker
---

<div align="center">

<h1>🤖 Transformer From Scratch — Educational Visualizer</h1>
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=6366F1&center=true&vCenter=true&width=700&lines=Train+a+Transformer+Step+by+Step;Live+Loss+Curves+%26+Attention+Heatmaps;English+%E2%86%92+French+Neural+Translation" alt="Typing SVG"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3b82f6?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-4f46e5?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-3b82f6?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Status](https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge)](#)

<br/>

[![Live Demo](https://img.shields.io/badge/%E2%96%B6_Live_Demo-Hugging_Face_Space-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/mnoorchenar/transformer-edu-viz)
[![Source Code](https://img.shields.io/badge/Source_Code-GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mnoorchenar/transformer-edu-viz)

<br/>

**🤖 Transformer From Scratch** — An interactive, step-by-step educational visualizer that trains a Transformer model from scratch for English→French translation, with live loss curves, real-time sample translations, and attention heatmaps — all inside your browser.

<br/>

---

</div>

## Table of Contents

- [Features](#-features)
- [Architecture](#️-architecture)
- [Getting Started](#-getting-started)
- [Docker Deployment](#-docker-deployment)
- [Dashboard Modules](#-dashboard-modules)
- [ML Models](#-ml-models)
- [Project Structure](#-project-structure)
- [Author](#-author)
- [Contributing](#-contributing)
- [Disclaimer](#️-disclaimer)
- [License](#-license)

---

## ✨ Features

| | Feature | Description |
|---|---|---|
| 📊 | **6-Step Wizard** | Guided pipeline: Data → Vocab → Model → Train → Evaluate → Infer |
| 📡 | **Live Training Stream** | Server-Sent Events stream real-time loss, LR, and translations per epoch |
| 🧠 | **Attention Heatmap** | Cross-attention weights visualized as an interactive color grid |
| 📈 | **Live Loss Chart** | Chart.js line chart updating in real-time with train and val loss |
| ⚙️ | **Configurable Model** | Adjust d_model, heads, layers, d_ff, dropout with live param count |
| 🔤 | **Greedy vs Beam** | Side-by-side comparison of decoding strategies with BLEU scores |

---

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

---

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

---

## 🐳 Docker Deployment

---

## 📊 Dashboard Modules

| Module | Description | Status |
|--------|-------------|--------|
| 📊 Data Explorer | Pair browser, augmentation config, dataset stats | ✅ Live |
| 📚 Vocabulary | Vocab builder with interactive tokenizer demo | ✅ Live |
| 🏗️ Model Config | Hyperparameter sliders with live parameter count | ✅ Live |
| 🎯 Live Training | SSE-driven loss chart, LR curve, live translations | ✅ Live |
| 📈 Evaluation | BLEU scores (greedy vs beam), translation table | ✅ Live |
| 🔤 Inference | Translate any sentence + cross-attention heatmap | ✅ Live |

---

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

---

## 📁 Project Structure

```
transformer-edu-viz/
│
├── 📂 templates/                 # HTML templates
│   └── index.html
├── 📂 transformer/
│   ├── data.py
│   └── model.py
├── 📄 Dockerfile                 # Container definition (port 7860)
├── 📄 __init__.py
├── 📄 app.py                     # Flask app, SSE stream, decode helpers
└── 📄 requirements.txt           # Python dependencies
```

---

## 👨‍💻 Author

<div align="center">

<table>
<tr>
<td align="center" width="100%">

<img src="https://avatars.githubusercontent.com/mnoorchenar" width="120" style="border-radius:50%; border: 3px solid #4f46e5;" alt="Mohammad Noorchenarboo"/>

<h3>Mohammad Noorchenarboo</h3>

<code>Data Scientist</code> &nbsp;|&nbsp; <code>AI Researcher</code> &nbsp;|&nbsp; <code>Biostatistician</code>

📍 &nbsp;Ontario, Canada &nbsp;&nbsp; 📧 &nbsp;[mohammadnoorchenarboo@gmail.com](mailto:mohammadnoorchenarboo@gmail.com)

──────────────────────────────────────

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mnoorchenar)&nbsp;
[![Personal Site](https://img.shields.io/badge/Website-mnoorchenar.github.io-4f46e5?style=for-the-badge&logo=githubpages&logoColor=white)](https://mnoorchenar.github.io/)&nbsp;
[![HuggingFace](https://img.shields.io/badge/HuggingFace-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/mnoorchenar)&nbsp;
[![Google Scholar](https://img.shields.io/badge/Scholar-4285F4?style=for-the-badge&logo=googlescholar&logoColor=white)](https://scholar.google.ca/citations?user=nn_Toq0AAAAJ&hl=en)&nbsp;
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mnoorchenar)

</td>
</tr>
</table>

</div>

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

---

## ⚠️ Disclaimer

> [!WARNING]
> This project is developed strictly for educational and research purposes. The dataset is hardcoded and synthetic. No real user data is stored. Provided "as is" without warranty of any kind.

---

## 📜 License

Distributed under the **MIT License**.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:4f46e5,100:06b6d4&height=120&section=footer&text=Made%20with%20%E2%9D%A4%EF%B8%8F%20by%20Mohammad%20Noorchenarboo&fontColor=ffffff&fontSize=18&fontAlignY=80" width="100%"/>

[![Star on GitHub](https://img.shields.io/badge/%E2%AD%90_Star-on_GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mnoorchenar/transformer-edu-viz/stargazers)
[![Fork on GitHub](https://img.shields.io/badge/%F0%9F%8D%B4_Fork-on_GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mnoorchenar/transformer-edu-viz/fork)

<sub>The name "Transformer From Scratch — Educational Visualizer" is used purely for academic and research purposes. Any similarity to existing product names or trademarks is entirely coincidental. This project has no affiliation with any commercial entity.</sub>

</div>
