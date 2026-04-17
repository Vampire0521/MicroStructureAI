<p align="center">
  <h1 align="center">🔬 MicroStructureAI</h1>
  <p align="center">
    <strong>CNN-Powered Steel Microstructure & Surface Defect Analyzer</strong>
  </p>
  <p align="center">
    <a href="#features">Features</a> •
    <a href="#demo">Demo</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#deployment">Deployment</a>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white" alt="FastAPI">
    <img src="https://img.shields.io/badge/Model-ResNet50-purple" alt="ResNet50">
    <img src="https://img.shields.io/badge/License-Educational-green" alt="License">
  </p>
</p>

---

A deep learning system that classifies **steel microstructure images** and **surface defects**, then generates complete metallurgical analysis reports — including heat treatment, composition, mechanical properties, root causes, remedies, and **Grad-CAM explainability visualizations**.

## Features

🔬 **Dual Analysis Modes**
- **Microstructure Mode** — Classifies UHCS micrographs into 6 phase categories
- **Surface Defect Mode** — Detects 6 types of NEU surface defects

🧠 **Intelligent Reporting**
- Heat treatment estimation
- Chemical composition range prediction
- Mechanical properties lookup (hardness, UTS, yield strength, elongation)
- Industrial applications
- Root cause analysis & recommended remedies (defects)

🔥 **Grad-CAM Explainability**
- Visual heatmaps showing which regions the CNN focuses on
- Original → Heatmap → Overlay comparison

⚠️ **Defect Flagging**
- Automatic alerts for problematic microstructures (Network Carbides, Widmanstätten)
- Severity ratings for surface defects
- Industry standard references (ASTM)

🌐 **Modern Web Interface**
- Drag-and-drop image upload
- Glassmorphism UI design
- Real-time analysis with loading animations
- Responsive layout

---

## Architecture

```
User uploads micrograph / surface image
        │
        ▼
┌─────────────────────────────────┐
│     Analysis Mode Selection      │
│   [Microstructure] [Surface]     │
└────────┬───────────────┬────────┘
         │               │
         ▼               ▼
┌────────────────┐ ┌────────────────┐
│  ResNet50 CNN  │ │  ResNet50 CNN  │
│  (UHCS-tuned)  │ │  (NEU-tuned)   │
│  6 classes     │ │  6 classes     │
└───────┬────────┘ └───────┬────────┘
        │                  │
        ▼                  ▼
   Predicted Class    Predicted Defect
   Confidence %       Severity Level
        │                  │
   ┌────┴────┐        ┌────┴────┐
   ▼         ▼        ▼         ▼
┌────────┐ ┌──────┐ ┌──────┐ ┌────────┐
│Grad-CAM│ │ UHCS │ │ NEU  │ │Grad-CAM│
│Heatmap │ │  KB  │ │  KB  │ │Heatmap │
└────────┘ └──────┘ └──────┘ └────────┘
        │                  │
        ▼                  ▼
┌──────────────────────────────────┐
│      Complete Analysis Report     │
│  • Classification + confidence   │
│  • Grad-CAM visualization        │
│  • Properties / Root causes      │
│  • Remedies / Applications       │
│  • Defect flags & severity       │
└──────────────────────────────────┘
```

---

## Datasets

### UHCS (Ultra-High Carbon Steel) — Microstructure

| Class | Description |
|-------|-------------|
| Pearlite | Lamellar ferrite + cementite |
| Spheroidite | Globular cementite in ferrite matrix |
| Network Carbides | Grain boundary cementite network |
| Widmanstätten | Crystallographic ferrite/cementite plates |
| Pearlite + Spheroidite | Partially spheroidized pearlite |
| Spheroidite + Widmanstätten | Mixed spheroidal + plate carbides |

📥 Download from [Kaggle](https://www.kaggle.com/datasets/) or [NIST](https://materialsdata.nist.gov/handle/11256/940)

### NEU Surface Defect Dataset

| Class | Description |
|-------|-------------|
| Rolled-in Scale | Oxide scale pressed into surface |
| Patches | Non-uniform cooling patterns |
| Crazing | Network of fine surface cracks |
| Pitted Surface | Chemical attack cavities |
| Inclusion | Non-metallic particles on surface |
| Scratches | Mechanical score marks |

📥 Download from [NEU Surface Defect Database](http://faculty.neu.edu.cn/songkechen/en/zdylm/263265/list/)

---

## Project Structure

```
MicroStructureAI/
│
├── app.py                      # FastAPI web server (main entry point)
├── step1_data_setup.py         # Data download & verification
├── step2_eda.py                # Exploratory data analysis
├── step3_preprocessing.py      # Image preprocessing & train/val/test split
├── step4_model.py              # ResNet50 model definition + Grad-CAM
├── step5_train.py              # Training loop with class-weighted loss
├── step6_evaluate.py           # Evaluation metrics & confusion matrix
├── step7_inference.py          # CLI inference pipeline
├── colab_train.py              # Google Colab training script (GPU)
│
├── knowledge_base.py           # Metallurgical knowledge database (9 classes)
├── defect_knowledge_base.py    # Surface defect + microstructural defect KB
│
├── templates/
│   └── index.html              # Web app frontend (glassmorphism UI)
├── static/
│   └── style.css               # Web app styles
│
├── models/                     # Trained model weights (.pth) — see below
├── data/                       # Datasets (download separately)
├── outputs/                    # Generated plots & results
│
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker config (HuggingFace Spaces)
├── Procfile                    # Cloud deployment config
├── runtime.txt                 # Python version pin
└── README.md                   # This file
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### 1. Clone & Install

```bash
git clone https://github.com/Vampire0521/MicroStructureAI.git
cd MicroStructureAI
pip install -r requirements.txt
```

### 2. Download Datasets

Download the UHCS and/or NEU datasets and extract into `data/`:

```
data/
├── UHCS/           # UHCS micrographs organized by class
│   ├── pearlite/
│   ├── spheroidite/
│   ├── network/
│   └── ...
└── NEU/            # NEU surface defect images
    ├── crazing/
    ├── inclusion/
    └── ...
```

### 3. Preprocess

```bash
python step1_data_setup.py       # Verify dataset structure
python step2_eda.py              # Exploratory data analysis
python step3_preprocessing.py    # Resize, split, compute class weights
```

### 4. Train

```bash
# Local training (CPU ~20-30 min per model)
python step5_train.py

# Google Colab (GPU ~3 min per model) — recommended
# Upload colab_train.py + data/processed/ to Colab, then:
!python colab_train.py --dataset uhcs    # Microstructure model
!python colab_train.py --dataset neu     # Surface defect model
!python colab_train.py --dataset both    # Both models
```

### 5. Evaluate

```bash
python step6_evaluate.py    # Confusion matrix, per-class metrics, ROC curves
```

### 6. Run

```bash
# CLI inference
python step7_inference.py path/to/image.png                  # Microstructure
python step7_inference.py path/to/image.png --mode surface   # Surface defect

# Web app
python app.py
# → Open http://localhost:8000
```

---

## Model Details

| Property | Value |
|----------|-------|
| **Base Architecture** | ResNet50 (ImageNet V2 pretrained) |
| **Fine-tuned Layers** | `layer3` + `layer4` + custom FC head |
| **Classifier Head** | FC(2048→256) → ReLU → Dropout(0.3) → FC(256→N) |
| **Input Size** | 224 × 224 RGB |
| **Trainable Params** | ~10M / 25M total |
| **Loss Function** | CrossEntropyLoss with inverse-frequency class weights |
| **Optimizer** | Adam (lr=1e-4, weight_decay=1e-4) |
| **Scheduler** | ReduceLROnPlateau (patience=3) |
| **Augmentations** | Flip, rotation, affine, color jitter |

### Trained Model Weights

The trained `.pth` model files are too large for GitHub (>90 MB each).

**Download from Google Drive / Hugging Face:**
> ⚠️ *Links to be added after upload*

Place them in the `models/` directory:
```
models/
├── best_microstructure_model.pth    # UHCS classifier (~92 MB)
└── best_surface_defect_model.pth    # NEU classifier (~92 MB)
```

---

## Knowledge Bases

### Metallurgical KB (`knowledge_base.py`)

Maps each microstructure to:
- **Heat treatment** — how this microstructure was likely produced
- **Composition range** — carbon content and typical steel grades
- **Mechanical properties** — hardness (HRC), UTS, yield strength, elongation
- **Applications** — industrial uses for this material condition
- **Fun facts** — interesting metallurgical trivia

Sources: *Callister (Materials Science & Engineering)*, *ASM Handbooks Vol. 4 & 9*, *Avner (Intro to Physical Metallurgy)*

### Surface Defect KB (`defect_knowledge_base.py`)

Maps each defect to:
- **Severity level** — Low / Moderate / High
- **Root causes** — what went wrong in the process
- **Remedies** — corrective actions to fix the issue
- **Affected properties** — how the defect impacts the product
- **Industry standards** — ASTM references

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| ML Framework | PyTorch + torchvision |
| Model | ResNet50 (transfer learning) |
| Explainability | Grad-CAM |
| Web Backend | FastAPI + Uvicorn |
| Web Frontend | Vanilla HTML / CSS / JavaScript |
| Templating | Jinja2 |
| Image Processing | OpenCV, Pillow |
| Data Science | NumPy, scikit-learn, pandas, matplotlib, seaborn |

---

## Deployment

### Hugging Face Spaces (Recommended)

This project includes a `Dockerfile` ready for HF Spaces:

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Choose SDK: **Docker**
3. Push this repository (including model weights) to the Space
4. The app will be live at `https://huggingface.co/spaces/YOUR_USERNAME/MicroStructureAI`

### Render / Railway

Use the included `Procfile`:

```bash
web: uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Main web interface |
| `POST` | `/analyze` | Analyze an image (multipart form: `file` + `mode`) |
| `GET` | `/health` | Health check — shows model loading status |

### Example API Usage

```python
import requests

# Microstructure analysis
with open("micrograph.png", "rb") as f:
    response = requests.post(
        "http://localhost:8000/analyze",
        files={"file": f},
        data={"mode": "micro"}
    )
    result = response.json()
    print(f"Class: {result['kb_name']}")
    print(f"Confidence: {result['confidence']:.1%}")
```

---

## Contributing

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is for **educational and research purposes**. The UHCS dataset is provided by NIST, and the NEU dataset by Northeastern University, China.

---

<p align="center">
  Built with ❤️ using PyTorch & FastAPI
</p>
