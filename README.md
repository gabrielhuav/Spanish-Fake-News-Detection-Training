# Spanish Fake News Detection - Training Repository

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

This repository contains the unified dataset and training code for the DistilBERT model used in the Spanish fake news detection web application described in the academic paper:

**"Spanish Fake News Detection with Fine-Tuned DistilBERT Built upon a Unified Corpus for a Real-World Application"**

## 📋 Overview

This project presents a comprehensive framework for Spanish fake news detection, from corpus construction to the deployment of a functional web application. The repository includes:


- **Unified Corpus**: 61,674 Spanish news articles (49.8% fake, 50.2% real)
- **Optimized Model**: Fine-tuned DistilBERT with aggressive regularization strategy
- **Results**: 95.36% accuracy with effective overfitting control

## 🎯 Key Features

### Dataset
- **61,674 balanced articles** in Spanish
- Integration of **4 recognized academic datasets**
- **9,000 additional satirical articles** from "El Deforma"
- Near-perfect balance: 49.8% fake / 50.2% real
- Multiple domains: political, general, satirical
![corpus_balance](https://cdn-uploads.huggingface.co/production/uploads/68eed105e16d138f8b990711/10NnOqz5zn_VtkILJtk19.png)


### Model
- **Architecture**: `distilbert-base-multilingual-cased`
- **Accuracy**: 95.36% on test set
- **F1-Score**: 0.954 (macro)
- **Regularization strategy**:
  - Ultra-low learning rate: 5×10⁻⁶
  - Aggressive dropout: 0.7
  - Strong L2 regularization: 0.5
  - Manual weight decay: 0.02

### Improvement Over Classical Methods
- **+23.33 percentage points** over traditional metaheuristic algorithms
- Effective overfitting control: gap < 0.058
- Over **500 GPU hours** of systematic experimentation

## 📁 Repository Structure

```
.
├── corpus_unificado_es_deforma_completo.csv  # Complete dataset (61,674 articles)
├── training/
│   ├── distilbert_training_en.py             # Main training script
│   ├── final_metrics_v11.txt                  # Final model metrics
│   └── experiment_configuration_antioverfit_v11.txt
├── final_model_distilbert_es_antioverfit_v11/ # Trained model
│   ├── config.json
│   ├── tf_model.h5
│   └── tokenizer_config.json
└── README.md
```

## 🚀 Installation and Usage

### Prerequisites
```bash
Python >= 3.8
TensorFlow >= 2.10
transformers >= 4.30.0
CUDA-compatible GPU (recommended)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/gabrielhuav/Spanish-Fake-News-Detection-Training.git
cd Spanish-Fake-News-Detection-Training

# Install dependencies
pip install tensorflow transformers pandas numpy scikit-learn matplotlib seaborn keras-tuner
```

### Training the Model

```bash
# Run the training script
python training/distilbert_training_en.py
```

The script will:
1. Load and preprocess the corpus
2. Optimize hyperparameters with Keras Tuner
3. Final training with anti-overfitting strategy
4. Evaluation and metrics generation
5. Save the optimized model

### Using the Pre-trained Model

```python
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

# Load model and tokenizer
model = TFAutoModelForSequenceClassification.from_pretrained(
    "./final_model_distilbert_es_antioverfit_v11"
)
tokenizer = AutoTokenizer.from_pretrained(
    "./final_model_distilbert_es_antioverfit_v11"
)

# Make prediction
text = "Article title [SEP] Full article content..."
inputs = tokenizer(text, return_tensors="tf", truncation=True, max_length=128)
outputs = model(inputs)
prediction = tf.nn.softmax(outputs.logits, axis=-1)

# Interpret result
fake_prob = prediction[0][0].numpy()
real_prob = prediction[0][1].numpy()
print(f"FAKE: {fake_prob*100:.2f}% | REAL: {real_prob*100:.2f}%")
```

## 📊 Main Results

### Approach Comparison

| Method | Accuracy | F1-Score | Precision | Recall |
|--------|----------|----------|-----------|--------|
| **DistilBERT (V11)** | **95.36%** | **0.954** | **0.954** | **0.954** |
| Genetic Algorithm (GA) | 72.03% | 0.714 | 0.740 | 0.720 |
| Scatter Search (SS) | 67.64% | 0.669 | 0.693 | 0.676 |
| VNS | 66.78% | 0.659 | 0.686 | 0.667 |
| Simulated Annealing | 60.86% | 0.586 | 0.638 | 0.608 |
| PSO | 57.67% | 0.489 | 0.736 | 0.575 |

### Overfitting Control

- **Loss gap**: 0.058 (target: <0.04)
- **Accuracy gap**: 0.037
- **Completed epochs**: 23
- **Best epoch**: 17
- **Early stopping**: Enabled with patience of 8 epochs

## 🔬 Optimization Methodology

### Aggressive Regularization Strategy (V11)

1. **Ultra-low Learning Rate**: 5×10⁻⁶ to 8×10⁻⁷
2. **Aggressive Dropout**: 0.4 - 0.7
3. **L2 Regularization**: 0.05 - 0.5
4. **Manual Weight Decay**: 0.02 (2x stronger)
5. **Noise Injection**: 0.01 - 0.03
6. **Variable Batch Size**: 4 - 8
7. **LR Reduction Factor**: 0.15 (more aggressive)
8. **Early Stopping Patience**: 8 epochs

### Dataset Split

- **Training**: 70% (43,172 articles)
- **Validation**: 10% (6,167 articles)
- **Test**: 20% (12,335 articles)

## 📖 Corpus Composition

| Source | Articles | Percentage | Year | Focus |
|--------|----------|------------|------|-------|
| Blanco-Fernández et al. | 57,231 | 92.8% | 2024 | Political |
| Tretiakov | 1,958 | 3.2% | 2022 | General |
| Posadas-Durán et al. | 971 | 1.6% | 2019 | Stylometric |
| Acosta (UPM) | 598 | 1.0% | 2019 | Manual verification |
| El Deforma (scraping) | 9,000 | 14.6% | 2025 | Satirical |

## 🌐 Web Application

This model is integrated into a Dockerized web application for real-time URL analysis. Application repository:

**[Spanish-Fake-News-Detection-Web-App](https://github.com/gabrielhuav/Spanish-Fake-News-Detection-Web-App)**

### Application Features
- Real-time URL analysis
- Automatic content extraction with BeautifulSoup
- Simple and accessible web interface
- Docker deployment (single command)
- REST API with Flask

### Citation (Pending)

```bibtex
@article{hurtado2025spanish,
  title={Spanish Fake News Detection with Fine-Tuned DistilBERT Built upon a Unified Corpus for a Real-World Application}
```
## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 🔗 Related Links

- **Web Application**: [Spanish-Fake-News-Detection-Web-App](https://github.com/gabrielhuav/Spanish-Fake-News-Detection-Web-App)
- **DistilBERT Documentation**: [Hugging Face](https://huggingface.co/docs/transformers/model_doc/distilbert)
- **Full Paper**: [Pending publication]

**Note**: This is an academic research project. The model and data are publicly available to encourage reproducibility and future research in Spanish misinformation detection.