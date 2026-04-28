# train_distilbeto_lightning.py - PyTorch Backend (Lightning AI Studio)
# DistilBETO Fine-Tuning: Physical Checkpoints, TQDM, Anti-OOM, and 5 Metrics

import os
import sys
import gc
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from tqdm import tqdm
from datetime import datetime

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ==============================================================================
# GLOBAL CONFIGURATION (LIGHTNING AI STUDIO)
# ==============================================================================
MODEL_NAME = 'dccuchile/distilbert-base-spanish-uncased'
MAX_LENGTH = 128

STUDIO_PATH = "/teamspace/studios/this_studio"
DATASET_PATH = f"{STUDIO_PATH}/corpus_unificado_3_clases_final.csv"
FINAL_MODEL_PATH = f"{STUDIO_PATH}/final_model_distilbeto_3_classes"
RESULTS_FILE = f"{STUDIO_PATH}/academic_results_distilbeto.txt"
CHECKPOINT_PATH = f"{STUDIO_PATH}/checkpoint_distilbeto.pt" 

TUNING_TRIALS = 3
TUNING_EPOCHS = 2
FINAL_TRAINING_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# DATASET CLASS
# ==============================================================================
class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# ==============================================================================
# PHASE 1: DATA LOADING AND PREPARATION
# ==============================================================================
def load_and_prepare_data(batch_size=16):
    print("--- Phase 1: Loading and Processing Corpus ---")

    try:
        df = pd.read_csv(DATASET_PATH, sep=';')
    except FileNotFoundError:
        print(f"CRITICAL ERROR: File '{DATASET_PATH}' not found.")
        print("Please ensure the CSV file is uploaded to your Studio storage.")
        sys.exit(1)

    df.dropna(subset=['text', 'label'], inplace=True)
    df['label'] = df['label'].astype(int)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    if 'title' in df.columns:
        df['merged_text'] = df['title'].astype(str) + " [SEP] " + df['text'].astype(str)
    else:
        df['merged_text'] = df['text'].astype(str)

    X_train, X_temp, y_train, y_temp = train_test_split(
        df['merged_text'].tolist(), df['label'].tolist(),
        train_size=0.7, random_state=42, stratify=df['label']
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=2/3, random_state=42, stratify=y_temp
    )

    del df, X_temp, y_temp
    gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_enc = tokenizer(X_train, truncation=True, padding=True, max_length=MAX_LENGTH)
    val_enc   = tokenizer(X_val,   truncation=True, padding=True, max_length=MAX_LENGTH)
    test_enc  = tokenizer(X_test,  truncation=True, padding=True, max_length=MAX_LENGTH)

    train_loader = DataLoader(NewsDataset(train_enc, y_train), batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(NewsDataset(val_enc,   y_val),   batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(NewsDataset(test_enc,  y_test),  batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, y_test, tokenizer

# ==============================================================================
# SINGLE EPOCH TRAINING AND EVALUATION
# ==============================================================================
def train_one_epoch(model, loader, optimizer, scheduler, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch in tqdm(loader, desc="Training", leave=False):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        optimizer.zero_grad()

        with autocast():
            outputs = model(**batch)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == batch['labels']).sum().item()
        total += len(batch['labels'])

    return total_loss / len(loader), correct / total

def evaluate(model, loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with autocast():
                outputs = model(**batch)
            total_loss += outputs.loss.item()
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == batch['labels']).sum().item()
            total += len(batch['labels'])

    return total_loss / len(loader), correct / total

# ==============================================================================
# PHASE 2: HYPERPARAMETER TUNING
# ==============================================================================
def run_tuning(train_loader, val_loader):
    print("\n--- Phase 2: Hyperparameter Tuning via Optuna ---")
    def objective(trial):
        lr         = trial.suggest_categorical('lr',         [1e-5, 5e-6, 2e-6])
        dropout    = trial.suggest_categorical('dropout',    [0.3, 0.4])
        batch_size = trial.suggest_categorical('batch_size', [8, 16]) 

        # DistilBERT uses specific dropout configuration names
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=3, dropout=dropout, seq_classif_dropout=dropout, attention_dropout=dropout
        ).to(DEVICE)

        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(train_loader) * TUNING_EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
        scaler = GradScaler()

        best_val_loss = float('inf')
        for epoch in range(TUNING_EPOCHS):
            train_one_epoch(model, train_loader, optimizer, scheduler, scaler)
            val_loss, _ = evaluate(model, val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss

        del model
        torch.cuda.empty_cache()
        gc.collect()
        return best_val_loss

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=TUNING_TRIALS)
    print(f"\nBest hyperparameters found: {study.best_params}")
    return study.best_params

# ==============================================================================
# METRICS AND PLOTTING
# ==============================================================================
def calculate_academic_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    specificities = []
    for i in range(3):
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fp = np.sum(cm[:, i]) - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(spec)
    macro_specificity = np.mean(specificities)
    return acc, p, r, f1, macro_specificity

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['FAKE (0)', 'REAL (1)', 'SATIRE (2)'],
                yticklabels=['FAKE (0)', 'REAL (1)', 'SATIRE (2)'])
    plt.title('Confusion Matrix - DistilBETO', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.savefig(f'{STUDIO_PATH}/academic_confusion_matrix_distilbeto.png', dpi=300, bbox_inches='tight')
    plt.close()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    print("=" * 80)
    print("FINE-TUNING: DistilBETO (LIGHTNING AI STUDIO)")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("WARNING: No GPU detected. Ensure a GPU machine is active.")
        return
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, test_loader, y_test, tokenizer = load_and_prepare_data(batch_size=16)
    
    if train_loader is None:
        return

    if os.path.exists(CHECKPOINT_PATH):
        print("\n[INFO] Checkpoint found in Persistent Storage. Skipping Tuning Phase...")
        best_lr = 5e-6
        best_dropout = 0.3
        best_batch_size = 16
    else:
        best_params = run_tuning(train_loader, val_loader)
        best_lr = best_params['lr']
        best_dropout = best_params['dropout']
        best_batch_size = best_params['batch_size']
        
        # Reload with optimal batch size
        train_loader, val_loader, test_loader, y_test, tokenizer = load_and_prepare_data(batch_size=best_batch_size)
        
    print(f"\n--- Phase 3: Final Training ---")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, dropout=best_dropout, seq_classif_dropout=best_dropout, attention_dropout=best_dropout
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=best_lr, weight_decay=0.01)
    total_steps = len(train_loader) * FINAL_TRAINING_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    scaler = GradScaler()

    best_val_loss = float('inf')
    start_epoch = 0
    patience_counter = 0

    if os.path.exists(CHECKPOINT_PATH):
        print(f"\n>>>> RESUMING FROM CHECKPOINT: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
        print(f">>>> Resuming at Epoch {start_epoch + 1}. Best saved loss: {best_val_loss:.4f}\n")

    for epoch in range(start_epoch, FINAL_TRAINING_EPOCHS):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scheduler, scaler)
        v_loss, v_acc   = evaluate(model, val_loader)

        print(f"Epoch {epoch+1}/{FINAL_TRAINING_EPOCHS} | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | Val Loss: {v_loss:.4f} Acc: {v_acc:.4f}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_counter = 0
            
            print("   -> Improvement detected. Saving Checkpoint to Persistent Storage...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': v_loss,
            }, CHECKPOINT_PATH)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

    print("\n--- Phase 4: Final Academic Evaluation ---")
    
    # ANTI-OOM FIX: Delete heavy training components before loading the checkpoint
    del optimizer
    del scheduler
    del scaler
    gc.collect()
    torch.cuda.empty_cache()

    if os.path.exists(CHECKPOINT_PATH):
        print("Loading the best model from Checkpoint into CPU memory first to prevent OOM...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Clean up CPU memory after extraction
        del checkpoint
        gc.collect()
        torch.cuda.empty_cache()
    
    # Move the clean model back to GPU for fast evaluation
    model.to(DEVICE)
    model.eval()
    
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Final Testing"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with autocast():
                logits = model(**batch).logits
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())

    acc, p, r, f1, spec = calculate_academic_metrics(y_test, all_preds)
    
    academic_report = (
        "====================================================\n"
        "FINAL ACADEMIC METRICS (DistilBETO)\n"
        "====================================================\n"
        f"Overall Accuracy : {acc:.4f}\n"
        f"Macro Precision  : {p:.4f}\n"
        f"Macro Recall     : {r:.4f}\n"
        f"Macro F1-Score   : {f1:.4f}\n"
        f"Macro Specificity: {spec:.4f}\n"
        "====================================================\n"
        f"{classification_report(y_test, all_preds, target_names=['FAKE', 'REAL', 'SATIRE'])}\n"
    )

    print("\nFINAL RESULTS:")
    print(academic_report)

    with open(RESULTS_FILE, "w") as f:
        f.write(academic_report)

    plot_confusion_matrix(y_test, all_preds)

    try:
        model.save_pretrained(FINAL_MODEL_PATH)
        tokenizer.save_pretrained(FINAL_MODEL_PATH)
        print(f"\nFinal model successfully saved in: {FINAL_MODEL_PATH}")
    except Exception as e:
        print(f"Error saving the final model: {e}")

if __name__ == '__main__':
    main()