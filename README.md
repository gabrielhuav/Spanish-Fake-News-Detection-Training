# Systematic Fine-Tuning of Transformer Models for Domain-Specific Misinformation Detection in Spanish Social Media Text

![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Model](https://img.shields.io/badge/Model-BETO-green.svg)
![Dataset](https://img.shields.io/badge/🤗%20Dataset-USMSC-orange)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)
![Institution](https://img.shields.io/badge/Institution-UAM%20Azcapotzalco-red)

**Paper:** [Systematic Fine-Tuning of Transformer Models for Domain-Specific Misinformation Detection in Spanish Social Media Text](https://doi.org/10.3390/informatics13060083)  
**Authors:** Gabriel Hurtado Avilés, José A. Reyes-Ortiz*, Román A. Mora-Gutiérrez, Josué Padilla Cuevas, Óscar Herrera Alcántara  
**Institution:** Department of Systems, Autonomous Metropolitan University (UAM), Azcapotzalco Unit, Mexico City, Mexico

---

## 📌 Abstract

While social media platforms are primary vectors for misinformation, automated detection systems remain largely confined to English. This paper presents a transferable, three-stage framework for fine-tuning transformer models to detect domain-specific deceptive content in Spanish. The pipeline comprises:

1. **Corpus Unification** — merging fragmented datasets into a 61,674-article resource mapped into three classes (Real, Fake, Satire) to prevent stylistic confounding.
2. **Systematic Model Optimization** — benchmarking classical metaheuristics against eight transformer architectures (mBERT, XLM-RoBERTa, BETO, and others) using strong regularization to prevent overfitting.
3. **Production Deployment** — encapsulating the optimized model as a containerized web application for real-time inference.

The Spanish-specific BETO encoder achieved 89.18% overall accuracy and a near-perfect in-source F1-score on the satire class. However, a strict source-held-out test revealed that this satire performance is highly source-dependent: recall on satire from an unseen outlet drops to 0.08, indicating that single-source class construction leads the model to recognize the source rather than a generalizable category. We report this as a central methodological finding: **per-class corpus source diversity is the primary determinant of generalization**. The methodology is adaptable across domains — by substituting the training corpus, the same framework may in principle be retargeted to other digital threats (investment scams, phishing), provided that suitable labeled corpora are constructed and validated.

---

## 🗂️ Repository Structure

```text
.
├── corpus/                  # Unified Spanish Misinformation and Satire Corpus (USMSC)
├── training/                # Fine-tuning scripts for all 8 transformer architectures
├── baselines/               # Metaheuristic-optimized TF-IDF classifiers (GA, PSO, VNS, SS, MSA)
├── webapp/                  # Dockerized web application for real-time URL analysis
├── results/                 # Benchmark tables, confusion matrices, training curves
├── revision_round2/         # Round-2 revision: supporting experiments & reproducibility
└── README.md
```

---

## 📊 Dataset: Unified Spanish Misinformation and Satire Corpus (USMSC)

The corpus is the result of unifying and deduplicating 4 academic sources, reaching a total of 61,674 records after normalization and near-duplicate removal.

The figure below illustrates the full corpus construction pipeline — from raw heterogeneous sources through normalization, deduplication, and final class balancing:

<img width="1788" height="838" alt="Corpus construction pipeline" src="https://github.com/user-attachments/assets/581e4b21-6427-4439-af46-e130fe799289" />

| Class | Category | Count | Percentage |
| :---: | :--- | :--- | :--- |
| **0** | `FAKE` | 21,746 | 35.3% |
| **1** | `REAL` | 30,943 | 50.2% |
| **2** | `SATIRE` | 8,985 | 14.5% |
| **-** | **Total** | **61,674** | **100%** |

**Unified sources:**
* **Posadas-Durán et al. (2019)** — Detection of fake news in a new corpus for the Spanish language
* **Acosta (UPM, 2019)** — Construction of a News Dataset for the Training and Evaluation of Automated Classifiers
* **Tretiakov et al. (2022)** — Detection of false information in Spanish using machine learning techniques
* **Blanco-Fernández et al. (2024)** — Enhancing Misinformation Detection in Spanish Language with Deep Learning

**Design rationale:** Unlike binary (Fake/Real) formulations, the 3-class architecture addresses a construct validity problem identified in the literature: satirical content (parody) should not be conflated with malicious misinformation, as they carry distinct stylistic markers and communicative purposes. The resulting Deceptive + Satire structure (49.8%) almost perfectly balances the Real class (50.2%). Note, however, that the satire class is built from a single source (El Deforma); see the source-held-out analysis under `revision_round2/` for the important generalization caveat this entails.

🤗 **Dataset available at:** [gabrielhuav/Unified-and-Balanced-Spanish-Fake-News-Corpus](https://huggingface.co/datasets/gabrielhuav/Unified-and-Balanced-Spanish-Fake-News-Corpus)

---

## 🔬 Methodology: Three-Stage Framework

### Stage 1 — Corpus Unification
Four heterogeneous Spanish-language datasets were merged, normalized, and deduplicated (deduplication performed before the train/validation/test split, over the merged pool). Satire was treated as a standalone class to avoid stylistic confounding with malicious fake news.

### Stage 2a — Metaheuristic Baseline (Preliminary, Binary Classification)
Five classical machine learning algorithms (TF-IDF representations) optimized via metaheuristics were evaluated as a preliminary baseline on the binary corpus (Real vs. Fake). These binary results are a preliminary sanity check and are not directly comparable to the three-class benchmark below:

| Algorithm | Accuracy (%) | Macro F1 | Ranking |
| :--- | :---: | :---: | :---: |
| Genetic Algorithm (GA) | 72.03 | 0.714 | 1st |
| Scatter Search (SS) | 67.64 | 0.669 | 2nd |
| VNS | 66.78 | 0.659 | 3rd |
| Simulated Annealing (MSA) | 60.86 | 0.586 | 4th |
| Particle Swarm Optimization (PSO) | 57.67 | 0.489 | 5th |
| DistilBERT (binary reference) | 95.36 | 0.954 | — |

On this preliminary binary task, the DistilBERT reference outperformed the best classical algorithm (GA) by +23.33 percentage points, motivating the move to transformer fine-tuning. The task was then reformulated into the methodologically sound three-class setting.

### Stage 2b — Systematic Transformer Benchmarking (3-Class)
Eight transformer architectures were benchmarked under identical protocols (same fixed stratified split, seed 42) on the three-class corpus:

| Model Architecture | Accuracy | Macro F1 | Macro Precision | Macro Recall | Macro Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BETO (V11 Final) 🏆** | **0.8898** | **0.9095** | **0.9129** | **0.9070** | **0.9331** |
| XLM-RoBERTa-Large | 0.8843 | 0.9061 | 0.9069 | 0.9053 | 0.9311 |
| RoBERTa-Large-BNE | 0.8837 | 0.9038 | 0.9081 | 0.9006 | 0.9293 |
| XLM-RoBERTa-Base | 0.8825 | 0.9025 | 0.9095 | 0.8984 | 0.9274 |
| DistilBERT Multilingual | 0.8795 | 0.9012 | 0.9027 | 0.9000 | 0.9280 |
| mBERT | 0.8730 | 0.8962 | 0.8973 | 0.8953 | 0.9242 |
| DistilBETO | 0.8586 | 0.8819 | 0.8889 | 0.8763 | 0.9141 |
| RoBERTa-Base-BNE | 0.6402 | 0.5791 | 0.6581 | 0.6651 | 0.7667 |

**Key finding:** BETO, a monolingual Spanish encoder, outperformed all multilingual alternatives, confirming that monolingual pretraining more effectively captures the morphological and syntactic nuances of Spanish deceptive language. BETO achieved a near-perfect in-source F1-score on the Satire class; the generalization limits of this score are analyzed in the source-held-out test under `revision_round2/`.

#### V11 Regularization Strategy (final 3-class BETO configuration)
The final configuration prioritizes genuine generalization through:

| Hyperparameter | Value |
| :--- | :--- |
| Learning rate | $1 \times 10^{-6}$ (ultra-low) |
| Dropout | 0.3 |
| L2 regularization | 0.01 |
| Gaussian weight-noise | 0.01 |
| Batch size | 8 |
| Max sequence length | 128 tokens |
| Early stopping patience | 6 epochs |
| Generalization gap (val − train loss) | 0.066 |

*Note on the preliminary phase:* An earlier exploratory phase on the binary task (DistilBERT reference) explored stronger settings (e.g., dropout up to 0.7, L2 up to 0.5); those values belong to that preliminary exploration and are not the final 3-class BETO configuration reported above. Hyperparameters were searched with RandomSearch / Optuna over a compact 3-trial grid.

Training ran for up to 30 epochs and was governed by an early stopping criterion that selected the checkpoint of minimum validation loss (epoch 27 in the local TensorFlow run; epoch 6 in the cloud PyTorch run). The validation loss stabilized into a plateau rather than diverging, the signature of successful regularization.

### Stage 3 — Production Deployment
The winning BETO model is deployed as a Dockerized microservice web application:
* **Measured end-to-end latency:** ~1161 ms per URL on commodity CPU hardware (6-core/12-thread AMD CPU, 32 GB RAM), dominated by network-bound web scraping rather than model inference. Full benchmark protocol and raw results are in the companion web-app repository under `revision_round2_latency/`.
* **Full pipeline:** Web scraping → HTML parsing → tokenization → forward-pass classification
* **Modular decoupling** of inference engine from frontend: adapting to a new domain primarily requires swapping the model weights and tokenizer, with limited changes to the serving infrastructure.

---

## 🛡️ Adversarial Robustness Testing

To probe potential source leakage (where a model might memorize publisher names rather than learning deceptive semantics), a dual-perturbation adversarial test was conducted on the full held-out test set (12,336 records):
* **NER Masking:** All Named Entities (Persons, Organizations, Locations) replaced with generic tags (e.g., `[ORG]`) using SpaCy `es_core_news_sm`.
* **Typographical Noise Injection:** Random character swaps and deletions at 5% probability per word.

| Condition | Overall Accuracy | Macro F1 | Satire F1 | Satire Recall |
| :--- | :---: | :---: | :---: | :---: |
| Standard (no perturbation) | 89.18% | 0.9095 | ~1.00 | — |
| NER Masking + Typo Injection | 63.40% | 0.71 | 0.96 | 0.92 |

The SATIRE class remained robust to entity masking (F1: 0.96), which suggests that the model uses syntactic and stylistic cues rather than specific entities. *Important caveat:* this test perturbs entities but keeps the source fixed, so it does not establish source independence. The stronger source-held-out test (see `revision_round2/`) shows that satire performance does not transfer to an unseen outlet — qualifying this adversarial result substantially.

---

## 🔄 Framework Transferability

A central design goal of this work is that the three-stage pipeline is adaptable across domains. By substituting the training corpus, the same framework could in principle be applied to the domains below. These are illustrative directions for future work and have not been empirically demonstrated in the current study; each would require a dedicated, ideally multi-source, labeled corpus:

| Domain | Description |
| :--- | :--- |
| **Investment Scam Detection** | Pages impersonating state-owned enterprises (e.g., PEMEX) promoting fabricated profit schemes |
| **Fraudulent E-Commerce Detection** | Fake retail pages with fabricated testimonials, countdown timers, and artificial urgency |
| **Phishing Content Detection** | Social media campaigns sharing linguistic features with misinformation |
| **Cross-lingual Extension** | Multilingual base models support 100+ languages; extending requires Stage 1 corpus curation |

---

## ⚙️ Reproducibility

Cross-framework validation was performed by training the optimal BETO configuration in both PyTorch/Optuna (cloud) and TensorFlow/Keras (local). Both runs reached an identical Macro F1-Score of 0.9095, with overall accuracies of 88.98% (PyTorch) and 89.18% (TensorFlow) — indicating that the reported performance is a property of the regularization recipe and data rather than a specific backend. A single training seed (42) was used; multi-seed runs with confidence intervals are identified as future work.

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
| :--- | :--- |
| 🤗 **Unified Corpus (USMSC)** | [HuggingFace Dataset](https://huggingface.co/datasets/gabrielhuav/Unified-and-Balanced-Spanish-Fake-News-Corpus) |
| 🧠 **Training Repository** | [GitHub — Training](https://github.com/gabrielhuav/Spanish-Fake-News-Detection-Training) |
| 🌐 **Web Application** | [GitHub — Web App](https://github.com/gabrielhuav/Spanish-Fake-News-Detection-Web-App) |

---

## 📜 Citation

If you use this corpus, framework, or web application in your research, please cite our paper published in *Informatics (MDPI)*:

**DOI:** [10.3390/informatics13060083](https://doi.org/10.3390/informatics13060083)

```bibtex
@Article{informatics13060083,
AUTHOR = {Avilés, Gabriel Hurtado and Reyes-Ortiz, José A. and Mora-Gutiérrez, Román A. and Cuevas, Josué Padilla and Alcántara, Óscar Herrera},
TITLE = {Systematic Fine-Tuning of Transformer Models for Domain-Specific Misinformation Detection in Spanish Social Media Text},
JOURNAL = {Informatics},
VOLUME = {13},
YEAR = {2026},
NUMBER = {6},
ARTICLE-NUMBER = {83},
URL = {https://www.mdpi.com/2227-9709/13/6/83},
ISSN = {2227-9709},
ABSTRACT = {While social media platforms are primary vectors for misinformation, automated detection systems remain largely confined to English. This paper presents a transferable, three-stage framework for fine-tuning transformer models to detect domain-specific deceptive content in Spanish. The pipeline comprises: (1) corpus unification, merging fragmented datasets into a 61,674-article resource mapped into three classes (Real, Fake, Satire) to prevent stylistic confounding; (2) systematic model optimization, extensively benchmarking classical metaheuristics against eight transformer architectures (including mBERT, XLM-RoBERTa, and BETO) using strong regularization to mitigate overfitting; and (3) production deployment, encapsulating the optimized model as a containerized web application for real-time inference. Through rigorous experimentation, the Spanish-specific BETO encoder emerged as the strongest model for this task, achieving 89.18% overall accuracy. The model attains a near-perfect in-source F1-score on the satire class; however, a strict source-held-out test reveals that this performance is highly source-dependent—recall on satire from an unseen outlet drops to 0.08—indicating that single-source class construction leads the model to recognize the source rather than a generalizable category. We report this finding as a central methodological result: corpus design, and in particular the source diversity of each class, is the primary determinant of whether the framework generalizes. Adversarial robustness tests using named-entity masking and typo injection provide complementary evidence on the model’s reliance on semantic versus surface cues. The methodology is designed to be adaptable across domains: by substituting the training corpus, the same framework may in principle be retargeted to other digital threats, such as investment scams and phishing, provided that suitable labeled corpora are constructed and validated for each new domain. The complete framework, dataset, and application are released as open-source resources to support reproducible research and practical countermeasures against online misinformation.},
DOI = {10.3390/informatics13060083}
}
```

---

## 👥 Authors & Contributions

| Author | Contributions |
| :--- | :--- |
| **Gabriel Hurtado Avilés** | Conceptualization, Investigation, Software, Writing — Original Draft |
| **José A. Reyes-Ortiz** *(Corresponding)* | Conceptualization, Funding Acquisition, Methodology, Project Administration, Supervision, Writing — Review & Editing |
| **Román A. Mora-Gutiérrez** | Data Curation, Methodology, Supervision, Validation, Visualization |
| **Josué Padilla Cuevas** | Formal Analysis, Validation, Visualization |
| **Óscar Herrera Alcántara** | Formal Analysis, Resources, Supervision, Validation |

---

## 🙏 Acknowledgments

The authors thank Universidad Autónoma Metropolitana, Unidad Azcapotzalco, and the Secretaría de Ciencia, Humanidades, Tecnología e Innovación del Gobierno de México (SECIHTI) for scholarship No. 4013730 (CVU: 1313870), which supported the development of this research.

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

## 🔁 Round-2 Revision — Supporting Experiments and Reproducibility

This section documents the materials added during the second round of review *(Informatics, MDPI; Manuscript ID informatics-4202664)*. Each item is mapped to the reviewer comment it addresses. All files are in `[revision_round2/](./revision_round2/)`.

### 1. Cross-source satire generalization — `revision_round2/cross_source_satire/`
Addresses the request for a source-held-out test of the SATIRE class.

| File | Description |
| :--- | :--- |
| `WebScraping_Satire_Corpora.ipynb` | Colab notebook documenting both satire scrapings: the 9,000 El Deforma articles used to build the training corpus, and the 1,000 El Mundo Today (Spain) articles used as the external, source-held-out evaluation set. |
| `colab.txt` | Public (view-access) link to the live Colab notebook above. |
| `elmundotoday_satire_eval.csv` | The 1,000 El Mundo Today satirical articles (`titulo`, `texto`, `label=2`, `fuente`, `url`). Used for evaluation only — not part of the training corpus. |
| `evaluate_cross_source_satire.py` | Loads the deployed BETO model and runs inference over the external set. |
| `cross_source_satire_results.json` | Raw result: in-source SATIRE recall 1.00 → out-of-source recall 0.08 (796/919 errors → FAKE, 123 → REAL; mean confidence 0.76). |

This experiment is the basis for the Cross-Source Generalization of the Satire Class analysis and the finding that per-class corpus source diversity is the primary determinant of generalization.

```bash
python evaluate_cross_source_satire.py \
    --model ./final_model_beto_v11_3_classes \
    --eval-csv ./elmundotoday_satire_eval.csv --insource-recall 1.00
```

### 2. Reproducibility of the final model — `revision_round2/reproducibility/`
Addresses configuration-harmonization and methods-detail requests.

| File | Description |
| :--- | :--- |
| `train_beto_v11.py` | Final 3-class BETO training (cloud; PyTorch + Optuna). |
| `train_beto_v11_laptop.py` | Final 3-class BETO training (local; TensorFlow + Keras Tuner). |
| `training_history_beto_v11.csv` | Per-epoch train/validation loss and accuracy (basis for the 0.066 gap and the epoch-27 checkpoint). |
| `academic_results_beto_v11.txt` | Final per-class precision/recall/F1 (accuracy 0.8918, Macro F1 0.9095). |
| `resultados salida cmd BETO aka bert-base.txt` | Full console log of the local TensorFlow run. |

### 3. Figures — `revision_round2/figures/`
Addresses figure/table consistency.

| File | Description |
| :--- | :--- |
| `beto_training_curves.png` | 30-epoch convergence of the final 3-class BETO model (stable plateau; gold marker = end of run; best checkpoint at epoch 27). |
| `academic_confusion_matrix_beto_v11.png` | Final 3-class confusion matrix (FAKE/REAL/SATIRE). |

*The deployment latency benchmark (also requested this round) is hosted in the companion web-application repository, `Spanish-Fake-News-Detection-Web-App`, under `revision_round2_latency/`.*