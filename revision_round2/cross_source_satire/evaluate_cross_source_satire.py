#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_cross_source_satire.py
===============================
Evalua el modelo BETO V11 YA ENTRENADO sobre satira de una FUENTE EXTERNA
(El Mundo Today), sin reentrenar nada. Responde a la objecion 1 del Reviewer 3:
demostrar que la clase SATIRE generaliza fuera de El Deforma.

Que reporta:
  - SATIRE recall sobre la fuente externa (% clasificado correctamente como satira)
  - A donde van los errores (cuantos a FAKE, cuantos a REAL)
  - Distribucion de confianza
  - Comparacion con el recall in-source (El Deforma) si lo proporcionas

Uso (Colab o local, junto al modelo):
    python evaluate_cross_source_satire.py \
        --model ./models/beto_v11_3_classes \
        --eval-csv .../elmundotoday_satire_eval.csv

NO reentrena. Solo inferencia. Minutos de computo.
"""

import argparse
import csv
import json
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

LABELS = {0: "FAKE", 1: "REAL", 2: "SATIRE"}
MAX_LENGTH = 128   # MISMO que en entrenamiento (train_beto_v11*.py)


def load_eval(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r.get("titulo", ""), r.get("texto", "")))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./models/beto_v11_3_classes")
    ap.add_argument("--eval-csv", required=True,
                    help="CSV de la fuente externa (label=2 SATIRE)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--insource-recall", type=float, default=None,
                    help="(opcional) recall SATIRE en El Deforma para comparar")
    args = ap.parse_args()

    print(f"[INFO] Cargando modelo: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = TFAutoModelForSequenceClassification.from_pretrained(args.model)

    data = load_eval(args.eval_csv)
    print(f"[INFO] Articulos a evaluar (todos SATIRE): {len(data)}\n")

    preds, confs = [], []
    B = args.batch_size
    for i in range(0, len(data), B):
        batch = data[i:i+B]
        texts = [f"{t} [SEP] {x}" for t, x in batch]
        enc = tok(texts, return_tensors="tf", truncation=True,
                  padding=True, max_length=MAX_LENGTH)
        logits = model(enc).logits
        probs = tf.nn.softmax(logits, axis=1).numpy()
        preds.extend(probs.argmax(axis=1).tolist())
        confs.extend(probs.max(axis=1).tolist())
        print(f"  procesados {min(i+B, len(data))}/{len(data)}")

    preds = np.array(preds)
    confs = np.array(confs)
    n = len(preds)

    # Todos los gold son SATIRE (clase 2)
    correct = int((preds == 2).sum())
    to_fake = int((preds == 0).sum())
    to_real = int((preds == 1).sum())
    recall = correct / n

    print("\n" + "=" * 60)
    print("CROSS-SOURCE SATIRE GENERALIZATION  (El Mundo Today)")
    print("=" * 60)
    print(f"N articulos             : {n}")
    print(f"Clasificados SATIRE     : {correct}  (recall = {recall:.4f})")
    print(f"  -> error a FAKE       : {to_fake}  ({to_fake/n*100:.1f}%)")
    print(f"  -> error a REAL       : {to_real}  ({to_real/n*100:.1f}%)")
    print(f"Confianza media (todas) : {confs.mean():.4f}")
    print(f"Confianza media aciertos: {confs[preds==2].mean():.4f}"
          if correct else "Confianza media aciertos: n/a")
    if args.insource_recall is not None:
        drop = args.insource_recall - recall
        print(f"\nRecall in-source (Deforma): {args.insource_recall:.4f}")
        print(f"Recall out-of-source      : {recall:.4f}")
        print(f"Caida (generalization gap): {drop:.4f}")
    print("=" * 60)

    out = {
        "source": "ElMundoToday",
        "n": n, "satire_recall": recall,
        "errors_to_fake": to_fake, "errors_to_real": to_real,
        "mean_confidence": float(confs.mean()),
        "insource_recall": args.insource_recall,
    }
    with open("cross_source_satire_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("[OK] Guardado: cross_source_satire_results.json")


if __name__ == "__main__":
    main()
