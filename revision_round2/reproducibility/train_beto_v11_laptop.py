"""
Fake News Detection in Spanish: V11 Anti-Overfitting Strategy
Model: BETO (dccuchile/bert-base-spanish-wwm-cased)
Description: Fine-tuning script implementing strict L2 regularization, 
dynamic weight decay, and Gaussian noise injection to prevent overfitting 
on satirical constraints. Includes tracking for 5 academic metrics.
"""

import os
import sys
import gc
import pandas as pd
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ==============================================================================
# ENVIRONMENT AND HARDWARE CONFIGURATION
# ==============================================================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("[INFO] GPU Memory dynamic growth enabled.")
    except RuntimeError as e:
        print(f"[ERROR] GPU Configuration failed: {e}")

# Note: mixed_float16 is intentionally disabled to prevent numeric overflow 
# (loss: inf) caused by the aggressive regularization of the V11 strategy.
print("[INFO] Float32 Precision enabled for mathematical stability.")

# ==============================================================================
# GLOBAL PARAMETERS
# ==============================================================================
MODEL_NAME = 'dccuchile/bert-base-spanish-wwm-cased' 
MAX_LENGTH = 128  
DATASET_PATH = "corpus_unificado_3_clases_final.csv"
RESULTS_FILE = "academic_results_beto_v11.txt"
FINAL_MODEL_PATH = "./final_model_beto_v11_3_classes"

TUNING_EPOCHS = 8  
FINAL_TRAINING_EPOCHS = 30  
EARLY_STOPPING_PATIENCE = 8  
REDUCE_LR_FACTOR = 0.15  
REDUCE_LR_PATIENCE = 1  

# ==============================================================================
# DATASET LOADING AND PREPARATION
# ==============================================================================
def load_and_prepare_data(batch_size=8):
    print("[INFO] Phase 1: Loading and Processing Corpus (3 Classes)")
    
    try:
        df = pd.read_csv(DATASET_PATH, sep=';')
    except FileNotFoundError:
        print(f"[CRITICAL] Dataset file '{DATASET_PATH}' not found.")
        sys.exit(1)

    df.dropna(subset=['text', 'label'], inplace=True)
    df['label'] = df['label'].astype(int)
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("\n--- Corpus Balance Analysis ---")
    label_counts = df['label'].value_counts()
    print(f"Total clean records: {len(df)}")
    print(f"FAKE (0): {label_counts.get(0, 0)} records ({label_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"REAL (1): {label_counts.get(1, 0)} records ({label_counts.get(1, 0)/len(df)*100:.1f}%)")
    print(f"SATIRE (2): {label_counts.get(2, 0)} records ({label_counts.get(2, 0)/len(df)*100:.1f}%)")
    print("-------------------------------")

    if 'title' in df.columns:
        df['merged_text'] = df['title'].astype(str) + " [SEP] " + df['text'].astype(str)
    else:
        df['merged_text'] = df['text'].astype(str)
    
    print(f"[INFO] Splitting data (70% Train, 10% Validation, 20% Test) with batch_size: {batch_size}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        df['merged_text'].tolist(), df['label'].tolist(),
        train_size=0.7, random_state=42, stratify=df['label']
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(2/3), random_state=42, stratify=y_temp 
    )

    del df, X_temp, y_temp
    gc.collect()

    print(f"[INFO] Loading tokenizer: '{MODEL_NAME}'")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=MAX_LENGTH)
    val_encodings = tokenizer(X_val, truncation=True, padding=True, max_length=MAX_LENGTH)
    test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=MAX_LENGTH)

    train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train)).shuffle(2000).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), y_val)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), y_test)).batch(batch_size)

    return train_dataset, val_dataset, test_dataset, y_test, tokenizer

# ==============================================================================
# MODEL ARCHITECTURE (V11 STRATEGY)
# ==============================================================================
def build_model_antioverfit_v11(hp):
    model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3, from_pt=True)
    
    hp_learning_rate = hp.Choice('learning_rate', values=[5e-6, 2e-6, 1e-6])
    hp_dropout_rate = hp.Choice('dropout_rate', values=[0.3, 0.4, 0.5])
    hp_l2_reg = hp.Choice('l2_regularization', values=[0.01, 0.02, 0.05])
    hp_noise_factor = hp.Choice('noise_factor', values=[0.005, 0.01])
    
    model.config.hidden_dropout_prob = hp_dropout_rate
    model.config.attention_probs_dropout_prob = hp_dropout_rate
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=hp_learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8
    )
    
    # Apply L2 Regularization
    for layer in model.layers:
        if hasattr(layer, 'kernel_regularizer') and layer.kernel_regularizer is None:
            layer.kernel_regularizer = tf.keras.regularizers.l2(hp_l2_reg)
    
    # Custom Weight Decay Callback
    class WeightDecayCallback(tf.keras.callbacks.Callback):
        def __init__(self, weight_decay=0.01):
            super().__init__()
            self.weight_decay = weight_decay
            
        def on_batch_end(self, batch, logs=None):
            for layer in self.model.layers:
                if hasattr(layer, 'kernel') and layer.kernel is not None:
                    layer.kernel.assign(layer.kernel * (1 - self.weight_decay * self.model.optimizer.learning_rate))
    
    model.weight_decay_callback = WeightDecayCallback(weight_decay=0.01)
    
    # Custom Gaussian Noise Injection Callback
    class NoiseInjectionCallback(tf.keras.callbacks.Callback):
        def __init__(self, noise_factor=0.01):
            super().__init__()
            self.noise_factor = noise_factor
            
        def on_train_batch_begin(self, batch, logs=None):
            if batch % 10 == 0:  
                for layer in self.model.layers:
                    if hasattr(layer, 'kernel') and layer.kernel is not None:
                        noise = tf.random.normal(layer.kernel.shape, mean=0.0, stddev=self.noise_factor)
                        layer.kernel.assign_add(noise * 0.0001) 
    
    model.noise_injection_callback = NoiseInjectionCallback(noise_factor=hp_noise_factor)
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    return model

# ==============================================================================
# ACADEMIC CALLBACKS
# ==============================================================================
class MonitorOverfitting(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        train_acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        
        loss_gap = val_loss - train_loss
        acc_gap = train_acc - val_acc
        
        print(f"\n[EPOCH {epoch + 1} METRICS]")
        print(f"   Loss Gap (val - train): {loss_gap:.4f}")
        print(f"   Accuracy Gap (train - val): {acc_gap:.4f}")

class AcademicConvergencePlotter(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.epochs, self.train_acc, self.val_acc = [], [], []
        self.train_loss, self.val_loss = [], []
        self.best_epoch = 0
        self.best_val_loss = np.Inf
        
    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch + 1)
        self.train_acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        
        if logs.get('val_loss') < self.best_val_loss:
            self.best_val_loss = logs.get('val_loss')
            self.best_epoch = epoch + 1
            
    def on_train_end(self, logs=None):
        self.plot_final_complete_analysis()

    def plot_final_complete_analysis(self):
        if len(self.epochs) < 3: return
            
        plt.style.use("seaborn-v0_8-whitegrid")
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], width_ratios=[1, 1, 1])
        
        ax_main_acc = fig.add_subplot(gs[0, :2])
        ax_gap_acc = fig.add_subplot(gs[0, 2])
        ax_gap_loss = fig.add_subplot(gs[1, 0])
        ax_convergence = fig.add_subplot(gs[1, 1])
        ax_stats = fig.add_subplot(gs[1, 2])
        
        ax_main_loss = ax_main_acc.twinx()
        
        ax_main_acc.plot(self.epochs, self.train_acc, 'b-o', label='Train Accuracy', linewidth=3, alpha=0.8)
        ax_main_acc.plot(self.epochs, self.val_acc, 'r-o', label='Val Accuracy', linewidth=3, alpha=0.8)
        ax_main_loss.plot(self.epochs, self.train_loss, 'b--s', label='Train Loss', linewidth=2, alpha=0.6)
        ax_main_loss.plot(self.epochs, self.val_loss, 'r--s', label='Val Loss', linewidth=2, alpha=0.6)
        
        if self.best_epoch <= len(self.epochs):
            best_idx = self.best_epoch - 1
            ax_main_acc.axvline(x=self.best_epoch, color='gold', linestyle=':', linewidth=3)
            ax_main_acc.scatter([self.best_epoch], [self.val_acc[best_idx]], color='gold', s=300, marker='*')
        
        ax_main_acc.set_title('BETO V11 Convergence (3 Classes)', fontsize=16, fontweight='bold')
        
        acc_gap = [t - v for t, v in zip(self.train_acc, self.val_acc)]
        loss_gap = [v - t for t, v in zip(self.train_loss, self.val_loss)]
        
        ax_gap_acc.plot(self.epochs, acc_gap, 'purple', marker='o')
        ax_gap_acc.axhline(y=0.04, color='orange', linestyle='--')
        ax_gap_acc.set_title('Accuracy Gap')
        
        ax_gap_loss.plot(self.epochs, loss_gap, 'orange', marker='s')
        ax_gap_loss.axhline(y=0.04, color='orange', linestyle='--')
        ax_gap_loss.set_title('Loss Gap')
        
        ax_convergence.text(0.5, 0.5, 'Analysis Complete', ha='center', va='center', fontsize=14, fontweight='bold')
        ax_convergence.set_axis_off()
        
        stats_text = f"V11 STATISTICS\n\nEpochs: {len(self.epochs)}\nBest: {self.best_epoch}\nVal Acc: {self.val_acc[-1]:.3f}"
        ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes, fontsize=11, verticalalignment='top')
        ax_stats.set_axis_off()
        
        plt.tight_layout()
        plt.savefig('academic_convergence_analysis_v11.png', dpi=300, bbox_inches='tight')
        plt.close()

# ==============================================================================
# METRICS EVALUATION
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
    return acc, p, r, f1, np.mean(specificities)

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['FAKE (0)', 'REAL (1)', 'SATIRE (2)'], 
                yticklabels=['FAKE (0)', 'REAL (1)', 'SATIRE (2)'])
    plt.title('Confusion Matrix - BETO (V11 Strategy)', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.savefig('academic_confusion_matrix_beto_v11.png', dpi=300, bbox_inches='tight')

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    print("=" * 80)
    print("FINE-TUNING INITIALIZED: BETO (V11 Strategy)")
    print("=" * 80)

    train_dataset, val_dataset, test_dataset, y_true_test, tokenizer = load_and_prepare_data(batch_size=8)
    if train_dataset is None: 
        return

    print("\n[INFO] Phase 2: Hyperparameter Tuning (V11)")
    tuner = kt.RandomSearch(
        build_model_antioverfit_v11,
        objective=kt.Objective('val_loss', direction='min'),
        max_trials=3,  
        directory='kt_beto_v11_dir',
        project_name='beto_v11_3_classes',
        overwrite=True
    )
    
    callbacks_tuning = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        MonitorOverfitting()
    ]
    tuner.search(train_dataset, epochs=TUNING_EPOCHS, validation_data=val_dataset, callbacks=callbacks_tuning)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    # Hardcoded optimal batch size for memory stability
    batch_size_optimal = 8
    
    train_dataset, val_dataset, test_dataset, y_true_test, _ = load_and_prepare_data(batch_size=batch_size_optimal)

    print("\n[INFO] Phase 3: Final Training (V11)")
    final_model = build_model_antioverfit_v11(best_hps)
    
    callbacks_final = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE, min_lr=1e-9),
        MonitorOverfitting(),
        AcademicConvergencePlotter(),
        final_model.weight_decay_callback,
        final_model.noise_injection_callback
    ]
    
    final_model.fit(train_dataset, epochs=FINAL_TRAINING_EPOCHS, validation_data=val_dataset, callbacks=callbacks_final, verbose=1)

    print("\n[INFO] Phase 4: Final Academic Evaluation")
    predicciones_logits = final_model.predict(test_dataset).logits
    predicciones_clases = np.argmax(predicciones_logits, axis=1)
    
    acc, p, r, f1, spec = calculate_academic_metrics(y_true_test, predicciones_clases)
    
    academic_report = (
        "====================================================\n"
        "FINAL ACADEMIC METRICS (BETO V11)\n"
        "====================================================\n"
        f"Overall Accuracy : {acc:.4f}\n"
        f"Macro Precision  : {p:.4f}\n"
        f"Macro Recall     : {r:.4f}\n"
        f"Macro F1-Score   : {f1:.4f}\n"
        f"Macro Specificity: {spec:.4f}\n"
        "====================================================\n"
        "Detailed Per-Class Report:\n"
        f"{classification_report(y_true_test, predicciones_clases, target_names=['FAKE', 'REAL', 'SATIRE'])}\n"
    )

    print(academic_report)
    with open(RESULTS_FILE, "w") as f: 
        f.write(academic_report)

    plot_confusion_matrix(y_true_test, predicciones_clases)

    try:
        final_model.save_pretrained(FINAL_MODEL_PATH)
        tokenizer.save_pretrained(FINAL_MODEL_PATH)
        print(f"\n[INFO] Model successfully saved in: {FINAL_MODEL_PATH}")
    except Exception as e: 
        print(f"[ERROR] Failed to save model: {e}")

if __name__ == '__main__':
    main()