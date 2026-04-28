# eval_final.py - Script de Evaluación Segura
import os
import gc
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# --- RUTAS ---
MODEL_NAME = 'FacebookAI/xlm-roberta-large'
MAX_LENGTH = 128
STUDIO_PATH = "/teamspace/studios/this_studio"
DATASET_PATH = f"{STUDIO_PATH}/corpus_unificado_3_clases_final.csv"
FINAL_MODEL_PATH = f"{STUDIO_PATH}/final_model_xlmr_large_3_classes"
RESULTS_FILE = f"{STUDIO_PATH}/academic_results_xlm_roberta_large.txt"
CHECKPOINT_PATH = f"{STUDIO_PATH}/checkpoint_xlm_roberta.pt" 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def load_test_data(batch_size=16):
    print("--- Cargando datos de prueba ---")
    df = pd.read_csv(DATASET_PATH, sep=';')
    df.dropna(subset=['text', 'label'], inplace=True)
    df['label'] = df['label'].astype(int)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    if 'title' in df.columns:
        df['merged_text'] = df['title'].astype(str) + " [SEP] " + df['text'].astype(str)
    else:
        df['merged_text'] = df['text'].astype(str)

    _, X_temp, _, y_temp = train_test_split(df['merged_text'].tolist(), df['label'].tolist(), train_size=0.7, random_state=42, stratify=df['label'])
    _, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42, stratify=y_temp)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    test_enc = tokenizer(X_test, truncation=True, padding=True, max_length=MAX_LENGTH)
    test_loader = DataLoader(NewsDataset(test_enc, y_test), batch_size=batch_size, shuffle=False, num_workers=4)

    return test_loader, y_test, tokenizer

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
    return acc, p, r, f1, np.mean(specificities)

def main():
    print("=" * 80)
    print("EVALUACIÓN FINAL: XLM-ROBERTA-LARGE (RECUPERACIÓN ANTI-OOM)")
    print("=" * 80)

    test_loader, y_test, tokenizer = load_test_data(batch_size=16)

    print("\n--- Cargando el modelo en CPU temporalmente para evitar Crash ---")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    
    # EL TRUCO MAGICO: map_location='cpu' carga el peso en la RAM normal primero
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Limpiamos basura pesada de la memoria
    del checkpoint
    gc.collect()
    torch.cuda.empty_cache()

    print("--- Transfiriendo modelo a la GPU ---")
    model.to(DEVICE)
    model.eval()

    all_preds = []
    print("\n--- Evaluando Test Set ---")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing final"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with autocast():
                logits = model(**batch).logits
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())

    acc, p, r, f1, spec = calculate_academic_metrics(y_test, all_preds)
    
    academic_report = (
        "====================================================\n"
        "FINAL ACADEMIC METRICS (XLM-RoBERTa-Large)\n"
        "====================================================\n"
        f"Overall Accuracy : {acc:.4f}\n"
        f"Macro Precision  : {p:.4f}\n"
        f"Macro Recall     : {r:.4f}\n"
        f"Macro F1-Score   : {f1:.4f}\n"
        f"Macro Specificity: {spec:.4f}\n"
        "====================================================\n"
        f"{classification_report(y_test, all_preds, target_names=['FAKE', 'REAL', 'SATIRE'])}\n"
    )

    print("\nRESULTADOS FINALES:")
    print(academic_report)

    with open(RESULTS_FILE, "w") as f:
        f.write(academic_report)

    print("\n--- Generando Matriz de Confusión ---")
    cm = confusion_matrix(y_test, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['FAKE (0)', 'REAL (1)', 'SATIRE (2)'],
                yticklabels=['FAKE (0)', 'REAL (1)', 'SATIRE (2)'])
    plt.title('Confusion Matrix - XLM-RoBERTa-Large', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.savefig(f'{STUDIO_PATH}/academic_confusion_matrix_xlmr_large.png', dpi=300, bbox_inches='tight')

    print(f"\n¡Éxito! Resultados guardados en {RESULTS_FILE}")

if __name__ == '__main__':
    main()