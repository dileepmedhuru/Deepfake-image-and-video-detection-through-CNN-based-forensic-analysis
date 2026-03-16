"""
Model Evaluation Script
Evaluate trained EfficientNetB0 model on the test split.

Can be run from ANY directory — paths are resolved relative to this file:
    python backend\evaluate_model.py          (from project root)
    python evaluate_model.py                  (from backend\ folder)
    python D:\...\backend\evaluate_model.py   (absolute path)

All three work correctly.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive — works without a display
import matplotlib.pyplot as plt
import seaborn as sns

# ── Resolve project root from this file's location ───────────────────────────
# This script lives at:  <project_root>/backend/evaluate_model.py
# So project root is one level up from __file__
THIS_DIR     = os.path.dirname(os.path.abspath(__file__))   # .../backend
PROJECT_ROOT = os.path.dirname(THIS_DIR)                    # .../Forgery_detection

# ── Configuration (all paths absolute — never depend on cwd) ─────────────────
MODEL_PATH    = os.path.join(PROJECT_ROOT, 'ml_models',        'cnn_model.h5')
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, 'processed_dataset', 'test')
OUTPUT_DIR    = THIS_DIR          # confusion_matrix.png saved into backend/
BATCH_SIZE    = 32
IMG_SIZE      = (224, 224)


def evaluate_model(model_path, test_data_dir, batch_size=32):

    print("=" * 55)
    print("MODEL EVALUATION  —  EfficientNetB0 Deepfake Detector")
    print("=" * 55)
    print(f"\nProject root : {PROJECT_ROOT}")
    print(f"Model path   : {model_path}")
    print(f"Test data    : {test_data_dir}")

    # ── Load model ───────────────────────────────────────────────────────────
    print("\nLoading model...")
    if not os.path.exists(model_path):
        print(f"\n  ERROR: Model not found at:\n    {model_path}")
        print("  Run train_model.py first to generate cnn_model.h5")
        return None

    from tensorflow.keras.models import load_model
    model = load_model(model_path)
    print(f"  ✓ Model loaded")
    print(f"  Input  shape : {model.input_shape}")
    print(f"  Output shape : {model.output_shape}")

    # ── Verify test folder ────────────────────────────────────────────────────
    if not os.path.exists(test_data_dir):
        print(f"\n  ERROR: Test data not found at:\n    {test_data_dir}")
        print("  Run data_preprocessing.py first to generate processed_dataset/")
        return None

    subdirs = [d for d in os.listdir(test_data_dir)
               if os.path.isdir(os.path.join(test_data_dir, d))]
    print(f"\n  Test subfolders found: {subdirs}")
    for sd in subdirs:
        n = len(os.listdir(os.path.join(test_data_dir, sd)))
        print(f"    {sd}/  →  {n:,} files")

    # ── Data generator ────────────────────────────────────────────────────────
    # CRITICAL: use preprocess_input — NOT rescale=1./255
    # Training used preprocess_input from EfficientNet; evaluation must match.
    from tensorflow.keras.applications.efficientnet import preprocess_input
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    print("\nLoading test images...")
    test_datagen   = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    print(f"  Test samples : {test_generator.samples}")
    print(f"  Class map    : {test_generator.class_indices}")
    # class_indices is usually {'fake': 0, 'real': 1} (alphabetical)
    fake_label = test_generator.class_indices.get('fake', 0)
    real_label = test_generator.class_indices.get('real', 1)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\nEvaluating (this may take a minute)...")
    eval_results = model.evaluate(test_generator, verbose=1)

    if len(eval_results) == 4:
        test_loss, test_acc, test_prec, test_rec = eval_results
    elif len(eval_results) == 3:
        test_loss, test_acc, test_prec = eval_results
        test_rec = float('nan')
    else:
        test_loss, test_acc = eval_results[0], eval_results[1]
        test_prec = test_rec = float('nan')

    print("\n" + "=" * 55)
    print("TEST RESULTS")
    print("=" * 55)
    print(f"  Accuracy  : {test_acc  * 100:.2f}%")
    print(f"  Precision : {test_prec * 100:.2f}%")
    print(f"  Recall    : {test_rec  * 100:.2f}%")
    print(f"  Loss      : {test_loss:.4f}")

    # ── Predictions ───────────────────────────────────────────────────────────
    print("\nGenerating per-sample predictions...")
    test_generator.reset()
    raw_preds    = model.predict(test_generator, verbose=1)
    pred_scores  = raw_preds.flatten()
    pred_classes = (pred_scores > 0.5).astype(int)
    true_classes = test_generator.classes

    # ── Classification report ─────────────────────────────────────────────────
    from sklearn.metrics import classification_report, confusion_matrix

    # Build label names in the order matching class_indices
    if fake_label == 0:
        class_names = ['Fake', 'Real']
    else:
        class_names = ['Real', 'Fake']

    print("\n" + "=" * 55)
    print("CLASSIFICATION REPORT")
    print("=" * 55)
    print(classification_report(
        true_classes, pred_classes,
        target_names=class_names, digits=4
    ))

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(true_classes, pred_classes)
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(f"  Labels : {class_names}")
    print(cm)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix — EfficientNetB0 Deepfake Detector')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\n  ✓ Confusion matrix saved → {cm_path}")

    # ── Additional metrics ────────────────────────────────────────────────────
    # Always compute TP/TN/FP/FN with fake=positive class
    if fake_label == 0:
        tp, fn, fp, tn = cm.ravel()   # fake=0 → row0/col0 = TP
    else:
        tn, fp, fn, tp = cm.ravel()   # fake=1 → row1/col1 = TP

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = (2 * test_prec * test_rec / (test_prec + test_rec)
          if (test_prec + test_rec) > 0 else 0)

    print("\n" + "=" * 55)
    print("ADDITIONAL METRICS")
    print("=" * 55)
    print(f"  True Positives  (fake correctly flagged)  : {tp:,}")
    print(f"  True Negatives  (real correctly passed)   : {tn:,}")
    print(f"  False Positives (real wrongly flagged)    : {fp:,}")
    print(f"  False Negatives (fake missed)             : {fn:,}")
    print(f"  Specificity  (real recall)                : {specificity*100:.2f}%")
    print(f"  F1-Score                                  : {f1*100:.2f}%")

    # ── Score distribution ────────────────────────────────────────────────────
    real_mask = (true_classes == real_label)
    fake_mask = (true_classes == fake_label)
    print("\n" + "=" * 55)
    print("SCORE DISTRIBUTION  (model output 0=real → 1=fake)")
    print("=" * 55)
    print(f"  Real images avg score : {pred_scores[real_mask].mean():.4f}"
          f"  (ideal: close to 1.0  — real=class 1)")
    print(f"  Fake images avg score : {pred_scores[fake_mask].mean():.4f}"
          f"  (ideal: close to 0.0  — fake=class 0)")

    return {
        'accuracy':         test_acc,
        'precision':        test_prec,
        'recall':           test_rec,
        'f1_score':         f1,
        'specificity':      specificity,
        'confusion_matrix': cm,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
    }


if __name__ == "__main__":
    # Optional command-line overrides:
    #   python backend\evaluate_model.py  <model_path>  <test_data_dir>
    model_path    = sys.argv[1] if len(sys.argv) > 1 else MODEL_PATH
    test_data_dir = sys.argv[2] if len(sys.argv) > 2 else TEST_DATA_DIR

    results = evaluate_model(
        model_path=model_path,
        test_data_dir=test_data_dir,
        batch_size=BATCH_SIZE,
    )

    if results:
        print("\n" + "=" * 55)
        print("✓ Evaluation complete!")
        print(f"  Accuracy  : {results['accuracy']*100:.2f}%")
        print(f"  F1-Score  : {results['f1_score']*100:.2f}%")
        print("=" * 55)