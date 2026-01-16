# TextDeepFakeGuard

**Hybrid Detection of Synthetic Text with Stylometry, Transformers, and Robustness Analysis**

📌 *Research & applied project on detecting AI-generated (deepfake) text using classical ML, neural networks, transformers, ensembles, and explainability.*

---

## 🔍 Problem Statement
The development of large language models has led to the emergence of a large number of AI texts that need to be distinguished from human ones. This can be useful in education, journalism, social media, cybersecurity, and more. For this purpose, there are various methods of artificial intelligence and machine learning that allow, using different data, to detect such texts with varying degrees of effectiveness. This repository contains several models that demonstrate varying degrees of effectiveness depending on the set of texts and its size. Below is a readme.md file that will help with the configuration and use of these models.

**Goal:** build a robust system for detecting synthetic text that:
- works across languages (RU / EN),
- generalizes to unseen LLMs,
- remains stable under paraphrasing and style changes,
- is explainable and suitable for online deployment.

---
## 🧠 Methods Overview
We evaluate and compare multiple families of models:

### 1. Classical ML (Baselines)
- Logistic Regression
- Linear SVM
- Random Forest

**Features:**
- TF-IDF (word- and character-level n-grams)
- Stylometric features (lexical diversity, entropy, punctuation ratios, POS statistics)

---

### 2. Neural Networks
- Character-level CNN
- BiLSTM / GRU with attention

Used to capture sequential and morphological patterns typical for synthetic text.

---

### 3. Transformer Models (Fine-tuning)
- BERT
- RuBERT
- XLM-R
- mT5

Models are fine-tuned for binary classification (human vs synthetic text) and evaluated for generalization and inference speed.

---

## 🔥 Proposed Method: HSSE
### **Hybrid Stylometric–Semantic Ensemble**

The core contribution of this project is a hybrid ensemble approach combining complementary signals:

**Components:**
1. **Semantic probability** — output of a fine-tuned transformer
2. **Stylometric score** — classical ML on handcrafted linguistic features
3. **Perplexity gap** — difference between autoregressive LM and masked LM perplexity
4. **Stability score** — prediction variance under:
   - paraphrasing
   - back-translation
   - style transformation

**Final decision:** stacking via a meta-classifier (Logistic Regression / LightGBM).

This design improves robustness and interpretability while maintaining competitive accuracy.

---

## 🧪 Experimental Setup
- Data split: train / validation / test
- Cross-validation for classical models
- Evaluation on unseen LLM-generated texts

### Metrics
- Accuracy
- Precision / Recall
- F1-score
- ROC-AUC

### Additional Evaluations
- **Robustness score** (ΔF1 after text transformations)
- **Generalization gap** (seen vs unseen generators)
- **Latency** (ms per text)
- **Model size & memory footprint**

---

## 🔎 Explainability (XAI)
- SHAP values for ML and ensemble models
- Attention visualization for transformers
- Token-level importance heatmaps
- LIME for local explanations

---

## ⚡ Inference & Deployment
- Batch and real-time inference benchmarks
- REST API prototype
- Streamlit demo application

---

## 📁 Repository Structure
```text
text-deepfake-guard/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── features/
│   ├── stylometric.py
│   └── embeddings.py
│
├── models/
│   ├── ml/
│   ├── neural/
│   ├── transformers/
│   └── ensemble/
│
├── experiments/
│   ├── evaluation.ipynb
│   └── robustness.ipynb
│
├── xai/
│   └── shap_analysis.ipynb
│
├── inference/
│   ├── api.py
│   └── benchmark.py
│
├── demo/
│   └── streamlit_app.py
│
├── docs/
│   ├── methodology.pdf
│   └── results.md
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run
```bash
pip install -r requirements.txt
python inference/api.py
```

---

## 📌 Author
**Alexey Staroverov, Nikita Petrov**  
BSc Applied Mathematics, HSE University  
Interests: NLP, ML, AI Safety, Robustness, Explainability

---

## 📄 Citation
If you use this project in academic work:
```
@misc{textdeepfakeguard2025,
  title={TextDeepFakeGuard: Hybrid Detection of Synthetic Text},
  author={Staroverov Alexey, Petrov Nikita},
  year={2026}
}
```
