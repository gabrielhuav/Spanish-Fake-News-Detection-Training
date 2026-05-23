

---

## Round-2 Revision — Supporting Experiments and Reproducibility

This section documents the materials added during the second round of review of
the associated paper (Informatics, MDPI; Manuscript ID informatics-4202664).
Each item is mapped to the reviewer comment it addresses. All files are in
[`revision_round2/`](./revision_round2/).

### 1. Cross-source satire generalization — `revision_round2/cross_source_satire/`
*Addresses the request for a source-held-out test of the SATIRE class.*

| File | Description |
|------|-------------|
| `scrape_elmundotoday_colab.py` | Google Colab scraper that builds the external evaluation set from El Mundo Today (Spain), an outlet unseen during training. |
| `elmundotoday_satire_eval.csv` | 1,000 satirical articles (`titulo, texto, label=2, fuente, url`). **Used for evaluation only — not part of the training corpus.** |
| `evaluate_cross_source_satire.py` | Loads the deployed BETO model and runs inference over the external set. |
| `cross_source_satire_results.json` | Raw result: in-source SATIRE recall 1.00 → out-of-source recall **0.08** (796/919 errors → FAKE, 123 → REAL; mean confidence 0.76). |

This experiment supports the new subsection *Cross-Source Generalization of the
Satire Class* and the finding that per-class corpus source diversity is the
primary determinant of generalization.

```bash
python evaluate_cross_source_satire.py \
    --model ./final_model_beto_v11_3_classes \
    --eval-csv ./elmundotoday_satire_eval.csv --insource-recall 1.00
```

### 2. Reproducibility of the final model — `revision_round2/reproducibility/`
*Addresses the configuration-harmonization and methods-detail requests.*

| File | Description |
|------|-------------|
| `train_beto_v11.py` | Final 3-class BETO training (cloud; PyTorch + Optuna). |
| `train_beto_v11_laptop.py` | Final 3-class BETO training (local; TensorFlow + Keras Tuner). |
| `training_history_beto_v11.csv` | Per-epoch train/validation loss and accuracy (basis for the 0.066 generalization gap and the epoch-27 checkpoint selection). |
| `academic_results_beto_v11.txt` | Final per-class precision/recall/F1 (accuracy 0.8918, Macro F1 0.9095). |
| `resultados_salida_cmd_BETO.txt` | Full console log of the local TensorFlow run: search space, chosen hyperparameters, and the 30-epoch trajectory. |

Validated final configuration: learning rate 1×10⁻⁶, dropout 0.3, L2 0.01,
weight-noise 0.01, batch size 8, 128-token inputs; RandomSearch/Optuna over a
compact 3-trial grid. The same fixed stratified split (seed 42) was used for all
eight transformer architectures.

### 3. Figures — `revision_round2/figures/`
*Addresses the figure/table consistency request.*

| File | Description |
|------|-------------|
| `beto_training_curves.png` | 30-epoch convergence of the final 3-class BETO model (stable plateau; gold marker = end of run; best checkpoint at epoch 27). |
| `academic_confusion_matrix_beto_v11.png` | Final 3-class confusion matrix (FAKE/REAL/SATIRE), distinct from the preliminary binary matrices. |

> The deployment latency benchmark (also requested in this review round) is
> hosted in the companion web-application repository,
> `Spanish-Fake-News-Detection-Web-App`, under `revision_round2_latency/`.
