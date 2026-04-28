# Systematic Fine-Tuning of Transformer Models for Domain-Specific Misinformation Detection in Spanish Social Media Text
 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![BETO](https://img.shields.io/badge/Model-BETO-green.svg)](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗%20Dataset-USMSC-orange)](https://huggingface.co/datasets/gabrielhuav/Unified-and-Balanced-Spanish-Fake-News-Corpus)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)](https://www.docker.com/)
[![UAM](https://img.shields.io/badge/Institution-UAM%20Azcapotzalco-red)](https://www.azc.uam.mx/)
 
> **Paper:** *Systematic Fine-Tuning of Transformer Models for Domain-Specific Misinformation Detection in Spanish Social Media Text*  
> **Authors:** Gabriel Hurtado Avilés, José A. Reyes-Ortiz\*, Román A. Mora-Gutiérrez, Josué Padilla Cuevas, Óscar Herrera Alcántara  
> **Institution:** Department of Systems, Autonomous Metropolitan University (UAM), Azcapotzalco Unit, Mexico City, Mexico  
 
---
 
## 📌 Abstract
 
While social media platforms are primary vectors for misinformation, automated detection systems remain largely confined to English. This paper presents a **transferable, three-stage framework** for fine-tuning transformer models to detect domain-specific deceptive content in Spanish. The pipeline comprises:
 
1. **Corpus Unification** — merging fragmented datasets into a 61,674-article resource mapped into three classes (Real, Fake, Satire) to prevent stylistic confounding.
2. **Systematic Model Optimization** — benchmarking classical metaheuristics against eight transformer architectures (mBERT, XLM-RoBERTa, BETO, and others) using aggressive regularization to prevent overfitting.
3. **Production Deployment** — encapsulating the optimized model as a containerized web application for real-time inference.
 
The Spanish-specific BETO encoder achieved **89.18% overall accuracy** and a **perfect 1.0 F1-score** in isolating satire from malicious fake news. Adversarial robustness tests (NER masking + typo injection) confirmed that the model relies on deep semantic patterns rather than source memorization, maintaining a **0.96 F1-score on satirical content** even under severe text degradation. The methodology is **domain-agnostic**: by substituting the training corpus, the same framework can detect investment scams, phishing, and other digital threats.
 
---
 
## 🗂️ Repository Structure
 
```
.
├── corpus/                  # Unified Spanish Misinformation and Satire Corpus (USMSC)
├── training/                # Fine-tuning scripts for all 8 transformer architectures
├── baselines/               # Metaheuristic-optimized TF-IDF classifiers (GA, PSO, VNS, SS, MSA)
├── webapp/                  # Dockerized web application for real-time URL analysis
├── results/                 # Benchmark tables, confusion matrices, training curves
└── README.md
```
 
---
 
## 📊 Dataset: Unified Spanish Misinformation and Satire Corpus (USMSC)
 
The corpus is the result of unifying and deduplicating **4 academic sources**, reaching a total of **61,674 records** after normalization and near-duplicate removal.
 
The figure below illustrates the full corpus construction pipeline — from raw heterogeneous sources through normalization, deduplication, and final class balancing:
 
<img width="1788" height="838" alt="Corpus construction pipeline" src="https://github.com/user-attachments/assets/581e4b21-6427-4439-af46-e130fe799289" />
 
| Class | Category | Count | Percentage |
|:---:|:---|:---:|:---:|
| **0** | `FAKE` | 21,746 | 35.3% |
| **1** | `REAL` | 30,943 | 50.2% |
| **2** | `SATIRE` | 8,985 | 14.5% |
| **Total** | | **61,674** | **100%** |
 
**Unified sources:**
- Posadas-Durán et al. (2019) — *Detection of fake news in a new corpus for the Spanish language*
- Acosta (UPM, 2019) — *Construction of a News Dataset for the Training and Evaluation of Automated Classifiers*
- Tretiakov et al. (2022) — *Detection of false information in Spanish using machine learning techniques*
- Blanco-Fernández et al. (2024) — *Enhancing Misinformation Detection in Spanish Language with Deep Learning*
 
**Design rationale:** Unlike binary (Fake/Real) formulations, the 3-class architecture resolves a construct validity problem identified in the literature: **satirical content (parody) should not be confused with malicious misinformation**, as they carry distinct stylistic markers and communicative purposes. The resulting Deceptive + Satire structure (49.8%) almost perfectly balances the Real class (50.2%).
 
🤗 Dataset available at: [gabrielhuav/Unified-and-Balanced-Spanish-Fake-News-Corpus](https://huggingface.co/datasets/gabrielhuav/Unified-and-Balanced-Spanish-Fake-News-Corpus)
 
---
 
## 🔬 Methodology: Three-Stage Framework
 
### Stage 1 — Corpus Unification
Four heterogeneous Spanish-language datasets were merged, normalized, and deduplicated. Satire was treated as a standalone class to avoid stylistic confounding with malicious fake news.
 
### Stage 2a — Metaheuristic Baseline (Binary Classification)
Five classical machine learning algorithms (TF-IDF representations) optimized via metaheuristics were evaluated as a baseline on the binary corpus (Real vs. Fake):
 
| Algorithm | Accuracy (%) | Macro F1 | Ranking |
|:---|:---:|:---:|:---:|
| Genetic Algorithm (GA) | 72.03 | 0.714 | 1st |
| Scatter Search (SS) | 67.64 | 0.669 | 2nd |
| VNS | 66.78 | 0.659 | 3rd |
| Simulated Annealing (MSA) | 60.86 | 0.586 | 4th |
| Particle Swarm Optimization (PSO) | 57.67 | 0.489 | 5th |
| **DistilBERT (Transformer reference)** | **95.36** | **0.954** | **—** |
 
The DistilBERT model outperformed the best classical algorithm (GA) by **+23.33 percentage points**, demonstrating that TF-IDF representations cannot compensate for the lack of contextual semantic understanding — a gap referred to as the *semantic gap*.
 
### Stage 2b — Systematic Transformer Benchmarking (3-Class)
Eight transformer architectures were benchmarked under identical protocols using the **V11 aggressive regularization strategy** on the three-class corpus:
 
| Model Architecture | Accuracy | Macro F1 | Macro Precision | Macro Recall | Macro Specificity |
|:---|:---:|:---:|:---:|:---:|:---:|
| **BETO (V11 Final) 🏆** | **0.8898** | **0.9095** | **0.9129** | **0.9070** | **0.9331** |
| XLM-RoBERTa-Large | 0.8843 | 0.9061 | 0.9069 | 0.9053 | 0.9311 |
| RoBERTa-Large-BNE | 0.8837 | 0.9038 | 0.9081 | 0.9006 | 0.9293 |
| XLM-RoBERTa-Base | 0.8825 | 0.9025 | 0.9095 | 0.8984 | 0.9274 |
| DistilBERT Multilingual | 0.8795 | 0.9012 | 0.9027 | 0.9000 | 0.9280 |
| mBERT | 0.8730 | 0.8962 | 0.8973 | 0.8953 | 0.9242 |
| DistilBETO | 0.8586 | 0.8819 | 0.8889 | 0.8763 | 0.9141 |
| RoBERTa-Base-BNE | 0.6402 | 0.5791 | 0.6581 | 0.6651 | 0.7667 |
 
**Key finding:** BETO, a monolingual Spanish encoder, outperformed all multilingual alternatives, confirming that monolingual pretraining more effectively captures the morphological and syntactic nuances of Spanish deceptive language. Critically, BETO achieved a **perfect F1-score of 1.00 on the Satire class**, validating the decision to isolate satirical content as a distinct label.
 
### V11 Regularization Strategy
 
The final configuration prioritizes genuine generalization through:
 
| Hyperparameter | Value |
|:---|:---:|
| Learning rate | $5 \times 10^{-6}$ (ultra-low) |
| Dropout on classification head | 0.7 (aggressive) |
| L2 regularization on classification head | 0.5 |
| Early stopping patience | 8 epochs |
| Generalization gap (val − train loss) | 0.058 |
 
Training was dynamically governed by an **early stopping criterion**: training concluded autonomously when the validation loss plateaued for 8 consecutive passes, with optimal weights automatically restored from the best checkpoint. Because BETO aggressively extracts optimal semantic features within a single epoch, the Early Stopping callback successfully halted training before catastrophic overfitting could occur.
 
> Over **500 GPU hours** of experimentation were conducted across all architectures and hyperparameter configurations. The full optimization trajectory — including failed configurations — is documented to provide a reproducible recipe for practitioners.
 
### Stage 3 — Production Deployment
 
The winning BETO model is deployed as a **Dockerized microservice web application**:
- End-to-end inference latency: **< 450 ms per URL on standard CPU**
- Full pipeline: Web scraping → HTML parsing → tokenization → forward-pass classification
- Modular decoupling of inference engine from frontend enables continuous integration: retraining for a new domain only requires swapping the serialized model weights (`.pt` file), with **zero changes to the API or serving infrastructure**.
 
---
 
## 🛡️ Adversarial Robustness Testing
 
To directly address potential **source leakage** (where a model might memorize publisher names rather than learning deceptive semantics), a dual-perturbation adversarial test was conducted on the full held-out test set (12,336 records):
 
1. **NER Masking:** All Named Entities (Persons, Organizations, Locations) replaced with generic tags (e.g., `[ORG]`) using SpaCy `es_core_news_sm`.
2. **Typographical Noise Injection:** Random character swaps and deletions at 5% probability per word.
 
| Condition | Overall Accuracy | Macro F1 | Satire F1 | Satire Recall |
|:---|:---:|:---:|:---:|:---:|
| Standard (no perturbation) | 89.18% | 0.9095 | **1.00** | — |
| NER Masking + Typo Injection | 63.40% | 0.71 | **0.96** | 0.92 |
 
The SATIRE class remained remarkably robust (F1: 0.96), proving that the model identifies satire through **deep syntactic structures, irony markers, and stylistic exaggeration**, independently of the specific entities or political figures mentioned. The expected degradation in REAL/FAKE discrimination (FAKE F1: 0.60; REAL F1: 0.56) validates a core NLP principle: factual verification inherently relies on entity anchors.
 
---
 
## 🔄 Framework Transferability
 
A central contribution of this work is that the three-stage pipeline is **explicitly domain-agnostic**. By substituting the training corpus, the same framework can be applied to:
 
| Domain | Description |
|:---|:---|
| **Investment Scam Detection** | Pages impersonating state-owned enterprises (e.g., PEMEX) promoting fabricated profit schemes; combining authority impersonation with unrealistic financial promises |
| **Fraudulent E-Commerce Detection** | Fake retail pages with fabricated testimonials, countdown timers, and artificial urgency; liquidation sales impersonating major retail chains |
| **Phishing Content Detection** | Social media campaigns sharing linguistic features with misinformation: impersonation, urgency, and deceptive intent |
| **Cross-lingual Extension** | The `distilbert-base-multilingual-cased` baseline supports 100+ languages; extending to a new language requires only Stage 1 corpus curation |
 
---
 
## ⚙️ Reproducibility
 
Cross-framework validation was performed by retraining the optimal BETO configuration from scratch in both **PyTorch/Optuna** (cloud) and **TensorFlow/Keras** (local), yielding an **identical Macro F1-Score of 0.9095** in both environments. The V11 regularization strategy is fully **framework-agnostic**.
 
---
 
## 🚀 Quick Start
 
```bash
# Clone the repository
git clone https://github.com/gabrielhuav/Spanish-Fake-News-Detection-Training.git
cd Spanish-Fake-News-Detection-Training
 
# Install dependencies
pip install -r requirements.txt
 
# Run the web application via Docker
docker pull gabrielhuav/spanish-fakenews-detector
docker run -p 8000:8000 gabrielhuav/spanish-fakenews-detector
```
 
---
 
## 🔗 Resources
 
| Resource | Link |
|:---|:---|
| 🤗 Unified Corpus (USMSC) | [HuggingFace Dataset](https://huggingface.co/datasets/gabrielhuav/Unified-and-Balanced-Spanish-Fake-News-Corpus) |
| 🧠 Training Repository | [GitHub — Training](https://github.com/gabrielhuav/Spanish-Fake-News-Detection-Training) |
| 🌐 Web Application | [GitHub — Web App](https://github.com/gabrielhuav/Spanish-Fake-News-Detection-Web-App) |
 
---
 
## 📜 Citation
 
If you use this corpus, framework, or web application in your research, please cite:
 
Coming Soon...
 
---
 
## 👥 Authors & Contributions
 
| Author | Contributions |
|:---|:---|
| **Gabriel Hurtado Avilés** | Conceptualization, Investigation, Software, Writing — Original Draft |
| **José A. Reyes-Ortiz** *(Corresponding)* | Conceptualization, Funding Acquisition, Methodology, Project Administration, Supervision, Writing — Review & Editing |
| **Román A. Mora-Gutiérrez** | Data Curation, Methodology, Supervision, Validation, Visualization |
| **Josué Padilla Cuevas** | Formal Analysis, Validation, Visualization |
| **Óscar Herrera Alcántara** | Formal Analysis, Resources, Supervision, Validation |
 
---
 
## 🙏 Acknowledgments
 
The authors thank **Universidad Autónoma Metropolitana, Unidad Azcapotzalco**, and the *Secretaría de Ciencia, Humanidades, Tecnología e Innovación del Gobierno de México (SECIHTI)* for scholarship No. 4013730 (CVU: 1313870), which supported the development of this research.
 
---
 
## 📄 License
 
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
