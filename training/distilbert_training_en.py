import os
import gc
import json
import pandas as pd
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from tensorflow.keras import mixed_precision
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- Disable low-level logs ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ENABLE MIXED PRECISION
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# --- GLOBAL CONFIGURATION V11 CORRECTED: MAXIMUM REGULARIZATION WITHOUT LABEL_SMOOTHING ---
MODEL_NAME = 'distilbert-base-multilingual-cased' 
MAX_LENGTH = 128  # Keep reduced
DATASET_PATH = "corpus_unificado_es_deforma_completo.csv"
RESULTS_FILE = "evaluation_results_antioverfit_v11.txt"
CONFIG_FILE = "experiment_configuration_antioverfit_v11.txt"
METRICS_FILE = "final_metrics_v11.txt"
FINAL_MODEL_PATH = "./final_model_distilbert_es_antioverfit_v11"

# V11 CORRECTED CONFIGURATION: MAXIMUM REGULARIZATION FOR GAP < 0.04
TUNING_EPOCHS = 8  # Keep
FINAL_TRAINING_EPOCHS = 30  # Keep
EARLY_STOPPING_PATIENCE = 8  # INCREASED: More patience for better convergence
LR_REDUCTION_FACTOR = 0.15  # MORE AGGRESSIVE: 0.2 -> 0.15
REDUCE_LR_PATIENCE = 1  # KEEP: Fast like V6/V10

def load_and_prepare_data(batch_size=8):
    """Load, clean, analyze and prepare data with V11 strategy."""
    print("--- Phase 1: Loading and processing corpus (V11 MODE - MAXIMUM REGULARIZATION CORRECTED) ---")
    
    try:
        df = pd.read_csv(DATASET_PATH, sep=';')
    except FileNotFoundError:
        print(f"Error: File '{DATASET_PATH}' not found.")
        return None, None, None, None, None

    df.dropna(subset=['text', 'label'], inplace=True)
    df['label'] = df['label'].astype(int)
    
    # MORE AGGRESSIVE dataset shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("\n--- Corpus Balance Analysis ---")
    label_counts = df['label'].value_counts()
    print(f"Total clean records: {len(df)}")
    print(f"FAKE news (0): {label_counts.get(0, 0)} records ({label_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"REAL news (1): {label_counts.get(1, 0)} records ({label_counts.get(1, 0)/len(df)*100:.1f}%)")
    
    # Verify dataset balance
    balance_ratio = min(label_counts) / max(label_counts)
    if balance_ratio < 0.7:
        print(f"Warning: Imbalanced dataset detected (ratio: {balance_ratio:.2f})")
        print("This may contribute to overfitting - consider balancing techniques.")
    
    print("-------------------------------------------------")

    df['merged_text'] = df['title'].astype(str) + " [SEP] " + df['text'].astype(str)
    
    print("\nV11 Split: 70% training, 10% validation, 20% test...")
    print(f"Using batch_size: {batch_size}")

    # Split: 70% training, 10% validation, 20% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        df['merged_text'].tolist(), df['label'].tolist(),
        train_size=0.7, random_state=42, stratify=df['label']
    )
    # Split 30% temporary into 10% validation and 20% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(2/3), random_state=42, stratify=y_temp # 2/3 of 30% is 20%
    )
    print(f"Training set size: {len(X_train)} records ({len(X_train)/len(df)*100:.1f}%)")
    print(f"Validation set size: {len(X_val)} records ({len(X_val)/len(df)*100:.1f}%)")
    print(f"Test set size: {len(X_test)} records ({len(X_test)/len(df)*100:.1f}%)")

    del df, X_temp, y_temp
    gc.collect()

    print(f"\nLoading tokenizer for model: '{MODEL_NAME}'")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Tokenizing with MAX_LENGTH={MAX_LENGTH} (reduced to avoid overfitting)...")
    train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=MAX_LENGTH)
    val_encodings = tokenizer(X_val, truncation=True, padding=True, max_length=MAX_LENGTH)
    test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=MAX_LENGTH)

    # Datasets with MORE AGGRESSIVE shuffle
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train)).shuffle(2000).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), y_val)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), y_test)).batch(batch_size)

    return train_dataset, val_dataset, test_dataset, y_test, tokenizer

def build_model_antioverfit_v11(hp):
    """V11 CORRECTED: MAXIMUM REGULARIZATION for gap < 0.04 WITHOUT label_smoothing."""
    model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    # EVEN LOWER LEARNING RATES (goal: gap < 0.04)
    hp_learning_rate = hp.Choice('learning_rate', values=[5e-6, 2e-6, 1e-6, 8e-7])
    
    # MORE AGGRESSIVE DROPOUT (0.4-0.7 vs V10: 0.3-0.6)
    hp_dropout_rate = hp.Choice('dropout_rate', values=[0.4, 0.5, 0.6, 0.7])
    
    # STRONGER L2 REGULARIZATION (0.05-0.5 vs V10: 0.01-0.5)
    hp_l2_reg = hp.Choice('l2_regularization', values=[0.05, 0.1, 0.2, 0.5])
    
    # NEW: NOISE INJECTION for more regularization (replaces label_smoothing)
    hp_noise_factor = hp.Choice('noise_factor', values=[0.01, 0.02, 0.03])
    
    # NEW: SMALLER BATCH SIZE for more regularization
    hp_batch_size = hp.Choice('batch_size', values=[4, 6, 8])
    
    # Configure dropout in model
    model.config.hidden_dropout_prob = hp_dropout_rate
    model.config.attention_probs_dropout_prob = hp_dropout_rate
    
    # KEEP ADAM (without weight_decay)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=hp_learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8
    )
    print("V11 CORRECTED: Using Adam with MAXIMUM regularization (without label_smoothing)")
    
    # STRONGER L2 REGULARIZATION
    for layer in model.layers:
        if hasattr(layer, 'kernel_regularizer') and layer.kernel_regularizer is None:
            layer.kernel_regularizer = tf.keras.regularizers.l2(hp_l2_reg)
        # STRONGER Bias regularization
        if hasattr(layer, 'bias_regularizer') and layer.bias_regularizer is None:
            layer.bias_regularizer = tf.keras.regularizers.l2(hp_l2_reg * 0.2)  # 0.1 -> 0.2
        # STRONGER Activity regularization
        if hasattr(layer, 'activity_regularizer') and layer.activity_regularizer is None:
            layer.activity_regularizer = tf.keras.regularizers.l2(hp_l2_reg * 0.2)  # 0.1 -> 0.2
    
    # STRONGER MANUAL WEIGHT DECAY
    class WeightDecayCallback(tf.keras.callbacks.Callback):
        def __init__(self, weight_decay=0.02):  # 0.01 -> 0.02
            super().__init__()
            self.weight_decay = weight_decay
            
        def on_batch_end(self, batch, logs=None):
            for layer in self.model.layers:
                if hasattr(layer, 'kernel') and layer.kernel is not None:
                    layer.kernel.assign(layer.kernel * (1 - self.weight_decay * self.model.optimizer.learning_rate))
    
    model.weight_decay_callback = WeightDecayCallback(weight_decay=0.02)
    
    # NEW: NOISE INJECTION CALLBACK (replaces label_smoothing)
    class NoiseInjectionCallback(tf.keras.callbacks.Callback):
        def __init__(self, noise_factor=0.02):
            super().__init__()
            self.noise_factor = noise_factor
            
        def on_train_batch_begin(self, batch, logs=None):
            # Inject small noise into weights for additional regularization
            if batch % 10 == 0:  # Every 10 batches
                for layer in self.model.layers:
                    if hasattr(layer, 'kernel') and layer.kernel is not None:
                        noise = tf.random.normal(layer.kernel.shape, mean=0.0, stddev=self.noise_factor)
                        layer.kernel.assign_add(noise * 0.001)  # Very small noise
    
    model.noise_injection_callback = NoiseInjectionCallback(noise_factor=hp_noise_factor)
    
    # TRADITIONAL LOSS FUNCTION (without label_smoothing)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    return model

class MonitorOverfitting(tf.keras.callbacks.Callback):
    """V11 Monitor: Stricter alerts for gap < 0.04."""
    
    def on_epoch_end(self, epoch, logs=None):
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        train_acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        
        loss_gap = val_loss - train_loss
        acc_gap = train_acc - val_acc
        
        print(f"\nV11 Monitoring Epoch {epoch + 1}:")
        print(f"   Loss gap (val - train): {loss_gap:.4f}")
        print(f"   Accuracy gap (train - val): {acc_gap:.4f}")
        
        # V11 STRICTER ALERTS (goal < 0.04)
        if loss_gap > 0.07:
            print("CRITICAL ALERT V11: Gap > 0.07 (goal < 0.04)")
        elif loss_gap > 0.04:
            print("WARNING V11: Gap > 0.04 (outside goal)")
        elif loss_gap > 0.02:
            print("GOOD V11: Gap between 0.02-0.04 (goal achieved)")
        else:
            print("EXCELLENT V11: Gap < 0.02 (perfect convergence)")
        
        if acc_gap > 0.04:
            print("ALERT V11: Accuracy gap > 0.04")
        elif acc_gap > 0.02:
            print("GOOD V11: Accuracy gap between 0.02-0.04")
        else:
            print("EXCELLENT V11: Accuracy gap < 0.02")

class RealTimeGraphicsV11(tf.keras.callbacks.Callback):
    """V11 Graphics with specific gap analysis."""
    
    def __init__(self):
        super().__init__()
        self.epochs = []
        self.train_acc = []
        self.val_acc = []
        self.train_loss = []
        self.val_loss = []
        self.best_epoch = 0
        self.best_val_loss = np.Inf
        
    def on_epoch_end(self, epoch, logs=None):
        # Save metrics
        self.epochs.append(epoch + 1)
        self.train_acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        
        # Track best epoch
        current_val_loss = logs.get('val_loss')
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.best_epoch = epoch + 1
        
        # Generate updated graph each epoch
        self.plot_progress()
        
    def on_train_end(self, logs=None):
        """V11 final graph with gap analysis."""
        self.plot_final_complete_analysis()
        
    def plot_progress(self):
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        
        # Accuracy graph
        ax1.plot(self.epochs, self.train_acc, 'b-o', label='Training Accuracy', linewidth=2.5, markersize=7)
        ax1.plot(self.epochs, self.val_acc, 'r-o', label='Validation Accuracy', linewidth=2.5, markersize=7)
        
        # Mark best epoch
        if self.best_epoch <= len(self.epochs):
            best_idx = self.best_epoch - 1
            ax1.scatter([self.best_epoch], [self.val_acc[best_idx]], 
                       color='gold', s=200, marker='*', label=f'Best epoch ({self.best_epoch})', zorder=5)
        
        ax1.set_title(f'V11: Accuracy Evolution - Epoch {self.epochs[-1]} | Best: {self.best_epoch}', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(self.epochs[::max(1, len(self.epochs)//10)])
        
        # Accuracy gap
        if len(self.epochs) > 1:
            acc_gap = [t - v for t, v in zip(self.train_acc, self.val_acc)]
            ax1_gap = ax1.twinx()
            ax1_gap.plot(self.epochs, acc_gap, 'g--', alpha=0.7, linewidth=1.5, label='Gap (Train-Val)')
            ax1_gap.axhline(y=0.04, color='orange', linestyle=':', alpha=0.8, label='V11 Goal (<0.04)')
            ax1_gap.axhline(y=0.02, color='green', linestyle=':', alpha=0.8, label='Excellent (<0.02)')
            ax1_gap.axhline(y=0, color='black', linestyle='-', alpha=0.5, label='Perfect Convergence')
            ax1_gap.set_ylabel('Accuracy Gap', fontsize=10, color='green')
            ax1_gap.legend(loc='upper right', fontsize=8)
        
        # Loss graph
        ax2.plot(self.epochs, self.train_loss, 'b-o', label='Training Loss', linewidth=2.5, markersize=7)
        ax2.plot(self.epochs, self.val_loss, 'r-o', label='Validation Loss', linewidth=2.5, markersize=7)
        
        # Mark best epoch
        if self.best_epoch <= len(self.epochs):
            best_idx = self.best_epoch - 1
            ax2.scatter([self.best_epoch], [self.val_loss[best_idx]], 
                       color='gold', s=200, marker='*', label=f'Best val_loss ({self.best_epoch})', zorder=5)
        
        ax2.set_title(f'V11: Loss Evolution - Epoch {self.epochs[-1]} | Best: {self.best_epoch}', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(self.epochs[::max(1, len(self.epochs)//10)])
        
        # V11 REAL-TIME GAP ANALYSIS
        if len(self.epochs) >= 3:
            current_loss_gap = self.val_loss[-1] - self.train_loss[-1]
            if current_loss_gap < 0.02:
                status_text = 'Gap < 0.02\nEXCELLENT!'
                color = 'green'
            elif current_loss_gap < 0.04:
                status_text = f'Gap = {current_loss_gap:.3f}\nV11 GOAL ACHIEVED'
                color = 'lightgreen'
            elif current_loss_gap < 0.07:
                status_text = f'Gap = {current_loss_gap:.3f}\nOutside goal'
                color = 'orange'
            else:
                status_text = f'Gap = {current_loss_gap:.3f}\nVery high'
                color = 'red'
                
            ax2.text(0.02, 0.98, status_text, 
                    transform=ax2.transAxes, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.7), fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'convergence_curve_v11_epoch_{self.epochs[-1]:02d}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"V11 graph saved: convergence_curve_v11_epoch_{self.epochs[-1]:02d}.png")

    def plot_final_complete_analysis(self):
        """V11 final analysis with focus on goal gap."""
        if len(self.epochs) < 3:
            return
            
        plt.style.use("seaborn-v0_8-whitegrid")
        fig = plt.figure(figsize=(20, 12))
        
        # Layout: 2x3 with larger main graph
        gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], width_ratios=[1, 1, 1])
        
        # Main graph: Complete metrics
        ax_main = fig.add_subplot(gs[0, :2])
        
        # Analysis graphs
        ax_gap_acc = fig.add_subplot(gs[0, 2])
        ax_gap_loss = fig.add_subplot(gs[1, 0])
        ax_convergence = fig.add_subplot(gs[1, 1])
        ax_stats = fig.add_subplot(gs[1, 2])
        
        # MAIN GRAPH: Accuracy and Loss
        ax_main_acc = ax_main
        ax_main_loss = ax_main.twinx()
        
        # Accuracy
        line1 = ax_main_acc.plot(self.epochs, self.train_acc, 'b-o', label='Train Accuracy', 
                                linewidth=3, markersize=8, alpha=0.8)
        line2 = ax_main_acc.plot(self.epochs, self.val_acc, 'r-o', label='Val Accuracy', 
                                linewidth=3, markersize=8, alpha=0.8)
        
        # Loss
        line3 = ax_main_loss.plot(self.epochs, self.train_loss, 'b--s', label='Train Loss', 
                                 linewidth=2, markersize=6, alpha=0.6)
        line4 = ax_main_loss.plot(self.epochs, self.val_loss, 'r--s', label='Val Loss', 
                                 linewidth=2, markersize=6, alpha=0.6)
        
        # Mark best epoch
        if self.best_epoch <= len(self.epochs):
            best_idx = self.best_epoch - 1
            ax_main_acc.axvline(x=self.best_epoch, color='gold', linestyle=':', linewidth=3, alpha=0.8)
            ax_main_acc.scatter([self.best_epoch], [self.val_acc[best_idx]], 
                               color='gold', s=300, marker='*', zorder=10, 
                               label=f'Best Epoch ({self.best_epoch})')
        
        ax_main_acc.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax_main_acc.set_ylabel('Accuracy', fontsize=14, fontweight='bold', color='blue')
        ax_main_loss.set_ylabel('Loss', fontsize=14, fontweight='bold', color='red')
        
        # Combine legends
        lines = line1 + line2 + line3 + line4
        labels = [l.get_label() for l in lines]
        if self.best_epoch <= len(self.epochs):
            labels.append(f'Best Epoch ({self.best_epoch})')
        ax_main_acc.legend(lines + [ax_main_acc.collections[-1]] if self.best_epoch <= len(self.epochs) else lines, 
                          labels, loc='center left', bbox_to_anchor=(1.1, 0.5))
        
        ax_main_acc.grid(True, alpha=0.3)
        ax_main_acc.set_title('V11: Maximum Regularization for Gap < 0.04 (CORRECTED)', fontsize=16, fontweight='bold', pad=20)
        
        # GAP ANALYSIS
        acc_gap = [t - v for t, v in zip(self.train_acc, self.val_acc)]
        loss_gap = [v - t for t, v in zip(self.train_loss, self.val_loss)]
        
        # Accuracy gap
        ax_gap_acc.plot(self.epochs, acc_gap, 'purple', marker='o', linewidth=2, markersize=5)
        ax_gap_acc.axhline(y=0.04, color='orange', linestyle='--', alpha=0.8, label='V11 Goal')
        ax_gap_acc.axhline(y=0.02, color='green', linestyle='--', alpha=0.8, label='Excellent')
        ax_gap_acc.axhline(y=0, color='black', linestyle='-', alpha=0.5, label='Convergence')
        ax_gap_acc.fill_between(self.epochs, acc_gap, alpha=0.3, color='purple')
        ax_gap_acc.set_title('Accuracy Gap\n(Train - Val)', fontsize=12, fontweight='bold')
        ax_gap_acc.set_ylabel('Gap', fontsize=10)
        ax_gap_acc.grid(True, alpha=0.3)
        ax_gap_acc.legend(fontsize=8)
        
        # Loss gap
        ax_gap_loss.plot(self.epochs, loss_gap, 'orange', marker='s', linewidth=2, markersize=5)
        ax_gap_loss.axhline(y=0.04, color='orange', linestyle='--', alpha=0.8, label='V11 Goal')
        ax_gap_loss.axhline(y=0.02, color='green', linestyle='--', alpha=0.8, label='Excellent')
        ax_gap_loss.axhline(y=0, color='black', linestyle='-', alpha=0.5, label='Convergence')
        ax_gap_loss.fill_between(self.epochs, loss_gap, alpha=0.3, color='orange')
        ax_gap_loss.set_title('Loss Gap\n(Val - Train)', fontsize=12, fontweight='bold')
        ax_gap_loss.set_xlabel('Epoch', fontsize=10)
        ax_gap_loss.set_ylabel('Gap', fontsize=10)
        ax_gap_loss.grid(True, alpha=0.3)
        ax_gap_loss.legend(fontsize=8)
        
        # CONVERGENCE ANALYSIS
        # Calculate convergence trend
        if len(self.epochs) >= 5:
            final_acc_gap = acc_gap[-1]
            final_loss_gap = loss_gap[-1]
            
            if final_acc_gap < 0.02 and final_loss_gap < 0.02:
                convergence_status = "PERFECT\nCONVERGENCE"
                color = 'green'
            elif final_acc_gap < 0.04 and final_loss_gap < 0.04:
                convergence_status = "GOAL\nACHIEVED"
                color = 'lightgreen'
            elif final_acc_gap < 0.07 and final_loss_gap < 0.07:
                convergence_status = "IMPROVED\nBUT NOT GOAL"
                color = 'yellow'
            else:
                convergence_status = "NO\nIMPROVEMENT"
                color = 'red'
        else:
            convergence_status = "ANALYZING\nCONVERGENCE"
            color = 'lightblue'
        
        ax_convergence.text(0.5, 0.7, convergence_status, ha='center', va='center', 
                          transform=ax_convergence.transAxes, fontsize=14, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))
        ax_convergence.text(0.5, 0.3, f'V11: Without\nLabel Smoothing', ha='center', va='center', 
                          transform=ax_convergence.transAxes, fontsize=10)
        ax_convergence.set_xlim(0, 1)
        ax_convergence.set_ylim(0, 1)
        ax_convergence.axis('off')
        
        # Final statistics
        final_train_acc = self.train_acc[-1]
        final_val_acc = self.val_acc[-1]
        final_acc_gap = final_train_acc - final_val_acc
        final_loss_gap = self.val_loss[-1] - self.train_loss[-1]
        
        stats_text = f"""V11 STATISTICS
        
Epochs: {len(self.epochs)}
Best epoch: {self.best_epoch}
Val Acc final: {final_val_acc:.3f}
Train Acc final: {final_train_acc:.3f}
Acc Gap: {final_acc_gap:.3f}
Loss Gap: {final_loss_gap:.3f}

{'GOAL ACHIEVED' if final_loss_gap < 0.04 else 'IMPROVED' if final_loss_gap < 0.07 else 'REVIEW'}"""
        
        ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes, 
                     fontsize=10, verticalalignment='top', 
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        ax_stats.set_xlim(0, 1)
        ax_stats.set_ylim(0, 1)
        ax_stats.axis('off')
        
        plt.tight_layout()
        plt.savefig('complete_convergence_analysis_v11.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nCOMPLETE V11 ANALYSIS: complete_convergence_analysis_v11.png")
        print(f"V11 Goal: Gap < 0.04 {'ACHIEVED' if final_loss_gap < 0.04 else 'NOT ACHIEVED'}")
        print(f"Final gap: {final_loss_gap:.3f}")

# V11 CORRECTED CALLBACKS
def create_antioverfit_callbacks_v11():
    """V11 Callbacks: Maximum regularization without label_smoothing."""
    
    # Early stopping with more patience for better convergence
    early_stopping_v11 = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,  # 8 epochs
        restore_best_weights=True,
        verbose=1,
        min_delta=0.0005,  # Stricter
        mode='min'
    )
    
    # More aggressive LR reduction
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=LR_REDUCTION_FACTOR,  # 0.15 (more aggressive)
        patience=REDUCE_LR_PATIENCE,  # 1
        min_lr=1e-9,  # Lower
        verbose=1,
        min_delta=0.0005
    )
    
    monitor_overfit_v11 = MonitorOverfitting()
    graphics_v11 = RealTimeGraphicsV11()
    
    return [early_stopping_v11, reduce_lr, monitor_overfit_v11, graphics_v11]

def save_experiment_configuration(best_hps, train_size, val_size, test_size, batch_size_used):
    """Save V11 CORRECTED experiment configuration."""
    config = {
        "experiment_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "V11 - MAXIMUM REGULARIZATION (CORRECTED)",
        "version": "V11 (Goal: Gap < 0.04) - Without Label Smoothing",
        "main_objective": "Reduce loss gap from ~0.10 (V10) to < 0.04",
        "v11_corrections": [
            "CORRECTED: Removed label_smoothing (not supported in your TensorFlow)",
            "ADDED: Noise injection as alternative to label_smoothing",
            "ADDED: Variable batch size (4,6,8) for more regularization",
            "MAINTAINED: All other regularization improvements"
        ],
        "v11_improvements": [
            "Even lower learning rates: 5e-6 to 8e-7",
            "More aggressive dropout: 0.4-0.7 (vs V10: 0.3-0.6)",
            "Stronger L2 regularization: 0.05-0.5 (vs V10: 0.01-0.5)",
            "Added noise injection: 0.01-0.03 (replaces label_smoothing)",
            "Variable batch size: 4-8 (more regularization)",
            "Increased manual weight decay: 0.02 (vs V10: 0.01)",
            "Stronger bias/activity regularization: 0.2 (vs V10: 0.1)",
            "More aggressive LR reduction: 0.15 (vs V10: 0.2)",
            "Increased patience: 8 (vs V10: 6) for better convergence",
            "Specific monitoring for gap < 0.04"
        ],
        "model": MODEL_NAME,
        "dataset": DATASET_PATH,
        "data_configuration": {
            "max_length": MAX_LENGTH,
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
            "split_percentages": "70% / 10% / 20%"
        },
        "v11_configuration": {
            "tuning_epochs": TUNING_EPOCHS,
            "final_training_epochs": FINAL_TRAINING_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "reduce_lr_patience": REDUCE_LR_PATIENCE,
            "lr_reduction_factor": LR_REDUCTION_FACTOR,
            "restore_best_weights": True,
            "optimizer": "Traditional Adam with MAXIMUM regularization",
            "regularization": "Strong L2 + Aggressive Dropout + Noise injection + Manual weight decay 2x"
        },
        "v11_objectives": {
            "loss_gap_goal": "< 0.04",
            "loss_gap_excellent": "< 0.02",
            "accuracy_gap_goal": "< 0.04",
            "expected_convergence": "Loss lines much closer"
        },
        "best_hyperparameters": {
            "learning_rate": best_hps.get('learning_rate'),
            "dropout_rate": best_hps.get('dropout_rate'),
            "l2_regularization": best_hps.get('l2_regularization'),
            "noise_factor": best_hps.get('noise_factor'),
            "batch_size": best_hps.get('batch_size'),
            "batch_size_used": batch_size_used
        },
        "mixed_precision": mixed_precision.global_policy().name
    }
    
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"\nV11 CORRECTED configuration saved to: '{CONFIG_FILE}'")
    except Exception as e:
        print(f"\nError saving configuration: {e}")

def save_final_metrics(history, y_true, y_pred, final_loss_gap, final_acc_gap, best_epoch, completed_epochs):
    """Save final metrics to text file."""
    
    # Generate classification report
    report_dict = classification_report(y_true, y_pred, target_names=['FAKE (0)', 'REAL (1)'], output_dict=True)
    
    # Calculate final metrics
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    accuracy = report_dict['accuracy']
    
    # Best epoch metrics
    best_epoch_idx = best_epoch - 1
    best_train_acc = history.history['accuracy'][best_epoch_idx]
    best_val_acc = history.history['val_accuracy'][best_epoch_idx]
    best_train_loss = history.history['loss'][best_epoch_idx]
    best_val_loss = history.history['val_loss'][best_epoch_idx]
    best_loss_gap = best_val_loss - best_train_loss
    best_acc_gap = best_train_acc - best_val_acc
    
    # Prepare metrics text
    metrics_text = f"""
================================================================================
        V11 FINAL METRICS - MAXIMUM REGULARIZATION (CORRECTED)
================================================================================
Experiment Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Model: {MODEL_NAME}
Dataset: {DATASET_PATH}

--------------------------------------------------------------------------------
                            TRAINING SUMMARY
--------------------------------------------------------------------------------
Total Epochs Completed: {completed_epochs}
Best Epoch: {best_epoch}
Epochs After Best: {completed_epochs - best_epoch}

V11 Goal: Loss gap < 0.04
V11 Achievement: {'YES - GOAL ACHIEVED' if final_loss_gap < 0.04 else 'NO - GOAL NOT ACHIEVED'}
Improvement vs V10: {((0.10 - final_loss_gap)/0.10)*100:.1f}%

--------------------------------------------------------------------------------
                         FINAL EPOCH METRICS
--------------------------------------------------------------------------------
Training Accuracy:      {final_train_acc:.4f} ({final_train_acc*100:.2f}%)
Validation Accuracy:    {final_val_acc:.4f} ({final_val_acc*100:.2f}%)
Training Loss:          {final_train_loss:.4f}
Validation Loss:        {final_val_loss:.4f}

Loss Gap (Val - Train): {final_loss_gap:.4f}
Accuracy Gap (Train - Val): {final_acc_gap:.4f}

Gap Status:
  - Loss Gap: {'EXCELLENT (<0.02)' if final_loss_gap < 0.02 else 'GOOD (<0.04)' if final_loss_gap < 0.04 else 'IMPROVED (<0.07)' if final_loss_gap < 0.07 else 'HIGH (>0.07)'}
  - Accuracy Gap: {'EXCELLENT (<0.02)' if final_acc_gap < 0.02 else 'GOOD (<0.04)' if final_acc_gap < 0.04 else 'REVIEW (>0.04)'}

--------------------------------------------------------------------------------
                        BEST EPOCH METRICS
--------------------------------------------------------------------------------
Training Accuracy:      {best_train_acc:.4f} ({best_train_acc*100:.2f}%)
Validation Accuracy:    {best_val_acc:.4f} ({best_val_acc*100:.2f}%)
Training Loss:          {best_train_loss:.4f}
Validation Loss:        {best_val_loss:.4f}

Loss Gap (Val - Train): {best_loss_gap:.4f}
Accuracy Gap (Train - Val): {best_acc_gap:.4f}

--------------------------------------------------------------------------------
                        TEST SET EVALUATION
--------------------------------------------------------------------------------
Test Accuracy:          {accuracy:.4f} ({accuracy*100:.2f}%)

Classification Report:
                    Precision    Recall    F1-Score    Support

FAKE (0)              {report_dict['FAKE (0)']['precision']:.4f}      {report_dict['FAKE (0)']['recall']:.4f}      {report_dict['FAKE (0)']['f1-score']:.4f}      {int(report_dict['FAKE (0)']['support'])}
REAL (1)              {report_dict['REAL (1)']['precision']:.4f}      {report_dict['REAL (1)']['recall']:.4f}      {report_dict['REAL (1)']['f1-score']:.4f}      {int(report_dict['REAL (1)']['support'])}

Macro Avg             {report_dict['macro avg']['precision']:.4f}      {report_dict['macro avg']['recall']:.4f}      {report_dict['macro avg']['f1-score']:.4f}      {int(report_dict['macro avg']['support'])}
Weighted Avg          {report_dict['weighted avg']['precision']:.4f}      {report_dict['weighted avg']['recall']:.4f}      {report_dict['weighted avg']['f1-score']:.4f}      {int(report_dict['weighted avg']['support'])}

--------------------------------------------------------------------------------
                    V11 REGULARIZATION TECHNIQUES
--------------------------------------------------------------------------------
- Learning Rate: Very low (5e-6 to 8e-7)
- Dropout: Aggressive (0.4-0.7)
- L2 Regularization: Strong (0.05-0.5)
- Noise Injection: Active (0.01-0.03) - Replaces label_smoothing
- Variable Batch Size: 4-8
- Manual Weight Decay: 0.02 (2x stronger)
- Bias/Activity Regularization: 0.2 (2x stronger)
- LR Reduction Factor: 0.15 (more aggressive)
- Early Stopping Patience: 8 epochs

--------------------------------------------------------------------------------
                           CONVERGENCE ANALYSIS
--------------------------------------------------------------------------------
Final Convergence Status: {'PERFECT' if final_loss_gap < 0.02 and final_acc_gap < 0.02 else 'GOAL ACHIEVED' if final_loss_gap < 0.04 and final_acc_gap < 0.04 else 'IMPROVED' if final_loss_gap < 0.07 else 'NEEDS REVIEW'}

Overfitting Risk: {'LOW' if final_loss_gap < 0.04 else 'MODERATE' if final_loss_gap < 0.07 else 'HIGH'}

Training Stability: {'STABLE' if completed_epochs - best_epoch < 5 else 'MODERATE' if completed_epochs - best_epoch < 10 else 'UNSTABLE'}

--------------------------------------------------------------------------------
                              CONCLUSIONS
--------------------------------------------------------------------------------
V11 Objective (Gap < 0.04): {'ACHIEVED' if final_loss_gap < 0.04 else 'NOT ACHIEVED'}
Model Performance: {'EXCELLENT' if accuracy > 0.95 else 'GOOD' if accuracy > 0.90 else 'ACCEPTABLE' if accuracy > 0.85 else 'NEEDS IMPROVEMENT'}
Generalization: {'EXCELLENT' if final_loss_gap < 0.02 else 'GOOD' if final_loss_gap < 0.04 else 'MODERATE' if final_loss_gap < 0.07 else 'POOR'}

Recommended Actions:
{'- Model ready for production' if final_loss_gap < 0.04 and accuracy > 0.90 else '- Consider additional regularization techniques' if final_loss_gap >= 0.04 else '- Monitor performance on new data'}
{'- Excellent generalization achieved' if final_loss_gap < 0.02 else '- Good balance between accuracy and generalization' if final_loss_gap < 0.04 else '- May benefit from more training data or stronger regularization'}

================================================================================
                            END OF METRICS REPORT
================================================================================
"""
    
    try:
        with open(METRICS_FILE, 'w', encoding='utf-8') as f:
            f.write(metrics_text)
        print(f"\nFinal metrics saved to: '{METRICS_FILE}'")
    except Exception as e:
        print(f"\nError saving metrics: {e}")

def plot_confusion_matrix(y_true, y_pred):
    """Generate V11 confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['FAKE (0)', 'REAL (1)'], 
                yticklabels=['FAKE (0)', 'REAL (1)'],
                cbar_kws={'label': 'Number of Predictions'})
    plt.title('Confusion Matrix - V11 Model (Maximum Regularization CORRECTED)', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.savefig('confusion_matrix_v11.png', dpi=300, bbox_inches='tight')
    print("V11 confusion matrix saved as 'confusion_matrix_v11.png'")

def main():
    """V11 CORRECTED main function: MAXIMUM REGULARIZATION for gap < 0.04."""
    print("="*80)
    print("  V11 TRAINING - MAXIMUM REGULARIZATION (Goal: Gap < 0.04) - CORRECTED")
    print("="*80)
    print("V10 -> V11 ANALYSIS:")
    print(f"   V10 Observed gap: ~0.10 (HIGH category)")
    print(f"   V11 Goal: < 0.04 (GOOD category)")
    print(f"   Required reduction: ~60% of current gap")
    print("\nV11 CORRECTIONS:")
    print("   X Label smoothing removed (not supported)")
    print("   + Noise injection added (alternative)")
    print("   + Variable batch size (4,6,8)")
    print("\nV11 MAINTAINED STRATEGIES:")
    print("   - LOWER learning rates: 5e-6 to 8e-7")
    print("   - MORE AGGRESSIVE dropout: 0.4-0.7")
    print("   - STRONGER L2 regularization: Minimum 0.05")
    print("   - Manual weight decay 2X: 0.02")
    print("   - Bias/Activity regularization 2X: 0.2")
    print("   - More aggressive LR reduction: 0.15")
    print("   - Increased patience: 8 epochs")
    print("="*80)
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU detected: {gpus[0]}")
        print(f"Mixed Precision: {mixed_precision.global_policy().name}")
    else:
        print("Warning: No GPU detected - using CPU")

    # Load data
    train_dataset, val_dataset, test_dataset, y_true_test, tokenizer = load_and_prepare_data()
    if train_dataset is None: 
        return

    # Get sizes
    train_size = sum(1 for _ in train_dataset.unbatch())
    val_size = sum(1 for _ in val_dataset.unbatch())
    test_size = len(y_true_test)

    print(f"\n--- Phase 2: V11 Optimization (Maximum Regularization CORRECTED) ---")
    print(f"SPECIFIC GOAL: Loss gap < 0.04")
    print(f"Hyperparameters with maximum regularization (without label_smoothing)")
    
    # V11 CORRECTED Tuner
    tuner = kt.RandomSearch(
        build_model_antioverfit_v11,
        objective=kt.Objective('val_loss', direction='min'),
        max_trials=4,
        directory='kt_v11_max_regularization_corrected_dir',
        project_name='distilbert_v11_max_reg_corrected_en',
        overwrite=True
    )
    
    callbacks_tuning = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        MonitorOverfitting()
    ]
    
    print("Searching V11 hyperparameters (maximum regularization CORRECTED)...")
    tuner.search(train_dataset, epochs=TUNING_EPOCHS, validation_data=val_dataset, callbacks=callbacks_tuning)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"\nOptimal V11 CORRECTED hyperparameters found:")
    print(f"   - Learning rate: {best_hps.get('learning_rate')} (LOWER)")
    print(f"   - Dropout rate: {best_hps.get('dropout_rate')} (MORE AGGRESSIVE)")
    print(f"   - L2 regularization: {best_hps.get('l2_regularization')} (STRONGER)")
    print(f"   - Noise factor: {best_hps.get('noise_factor')} (NEW - replaces label_smoothing)")
    optimal_batch_size = best_hps.get('batch_size')
    print(f"   - Optimal batch size: {optimal_batch_size} (variable 4-8)")

    # Recreate datasets with optimal batch_size
    train_dataset, val_dataset, test_dataset, y_true_test, _ = load_and_prepare_data(
        batch_size=optimal_batch_size
    )

    # Save configuration
    save_experiment_configuration(best_hps, train_size, val_size, test_size, optimal_batch_size)

    print(f"\n--- Phase 3: V11 CORRECTED Final Training ---")
    print(f"GOAL: Loss gap < 0.04")
    print(f"Patience: {EARLY_STOPPING_PATIENCE} epochs")
    print("MAXIMUM regularization activated (without label_smoothing)")
    
    final_model = build_model_antioverfit_v11(best_hps)
    final_callbacks = create_antioverfit_callbacks_v11()
    
    # Add additional callbacks
    if hasattr(final_model, 'weight_decay_callback'):
        final_callbacks.append(final_model.weight_decay_callback)
        print("Manual weight decay 2X activated")
    
    if hasattr(final_model, 'noise_injection_callback'):
        final_callbacks.append(final_model.noise_injection_callback)
        print("Noise injection activated (replaces label_smoothing)")
    
    print(f"\nStarting V11 CORRECTED training...")
    print("V11 EXPECTATION:")
    print("   - Loss gap: < 0.04 (vs V10: ~0.10)")
    print("   - Improved convergence: Lines CLOSER")
    print("   - Accuracy/generalization trade-off")
    print("="*60)
    
    history = final_model.fit(
        train_dataset,
        epochs=FINAL_TRAINING_EPOCHS,
        validation_data=val_dataset,
        callbacks=final_callbacks,
        verbose=1
    )

    completed_epochs = len(history.history['loss'])
    print(f"\nV11 CORRECTED training completed: {completed_epochs} epochs")
    
    # V11 specific analysis
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_loss_gap = final_val_loss - final_train_loss
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_acc_gap = final_train_acc - final_val_acc
    
    best_epoch_idx = np.argmin(history.history['val_loss'])
    best_epoch = best_epoch_idx + 1
    
    print(f"\nV11 CORRECTED ANALYSIS - GOAL GAP < 0.04:")
    print(f"   Final loss gap: {final_loss_gap:.4f}")
    if final_loss_gap < 0.02:
        print(f"   RESULT: EXCELLENT (<0.02)")
    elif final_loss_gap < 0.04:
        print(f"   RESULT: GOAL ACHIEVED (<0.04)")
    elif final_loss_gap < 0.07:
        print(f"   RESULT: IMPROVED but not goal (<0.07)")
    else:
        print(f"   RESULT: No significant improvement (>0.07)")
    
    print(f"   Improvement vs V10: {0.10 - final_loss_gap:.3f} ({((0.10 - final_loss_gap)/0.10)*100:.1f}%)")
    print(f"   Best epoch: {best_epoch}")
    print(f"   Epochs post-best: {completed_epochs - best_epoch}")
    
    # Final evaluation
    print("\n--- Phase 4: V11 CORRECTED Evaluation ---")
    predictions_logits = final_model.predict(test_dataset).logits
    predictions_classes = np.argmax(predictions_logits, axis=1)
    
    report_dict = classification_report(y_true_test, predictions_classes, target_names=['FAKE (0)', 'REAL (1)'], output_dict=True)
    accuracy = report_dict['accuracy']
    
    print(f"\nV11 CORRECTED FINAL RESULTS:")
    print(f"   - Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"   - Goal gap: {'ACHIEVED' if final_loss_gap < 0.04 else 'NOT ACHIEVED'}")
    print(f"   - Convergence: {'EXCELLENT' if final_loss_gap < 0.02 else 'GOOD' if final_loss_gap < 0.04 else 'IMPROVED'}")
    print(f"   - Techniques used: Weight decay 2X + Noise injection + Variable batch + Strong L2")
    
    # Save final metrics to text file
    save_final_metrics(history, y_true_test, predictions_classes, final_loss_gap, final_acc_gap, best_epoch, completed_epochs)
    
    # Generate graphs
    print("\n--- Phase 5: V11 Graphics Generation ---")
    plot_confusion_matrix(y_true_test, predictions_classes)
    
    # Save model
    try:
        print(f"\n--- Phase 6: Saving V11 model ---")
        final_model.save_pretrained(FINAL_MODEL_PATH)
        tokenizer.save_pretrained(FINAL_MODEL_PATH)
        
        print("\n" + "="*80)
        print("V11 CORRECTED EXPERIMENT COMPLETED")
        print("="*80)
        print(f"V11 GOAL: Gap < 0.04")
        print(f"RESULT: Gap = {final_loss_gap:.4f}")
        print(f"STATUS: {'SUCCESS' if final_loss_gap < 0.04 else 'PARTIAL' if final_loss_gap < 0.07 else 'REVIEW'}")
        print(f"IMPROVEMENT vs V10: {((0.10 - final_loss_gap)/0.10)*100:.1f}%")
        print(f"Techniques: Without label_smoothing + Noise injection + Maximum regularization")
        print(f"\nMetrics exported to: {METRICS_FILE}")
        
    except Exception as e:
        print(f"Error saving V11 model: {e}")

if __name__ == '__main__':
    main()