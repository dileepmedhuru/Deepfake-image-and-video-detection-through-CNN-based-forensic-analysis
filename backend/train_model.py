"""
CNN Training Script – Deepfake Detection
Uses EfficientNetB0 transfer learning + class-weight balancing.

Usage:
    python train_model.py                          # defaults
    python train_model.py --epochs 30 --lr 0.0005
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt


# ── Model ─────────────────────────────────────────────────────────────────────

def create_model(input_shape=(224, 224, 3)):
    """
    EfficientNetB0 backbone (ImageNet pre-trained) + custom classifier head.
    Much better accuracy than a hand-built CNN for this task size.
    """
    base = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
    )
    # Freeze backbone initially; we'll fine-tune later
    base.trainable = False

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid'),   # binary
    ])
    return model, base


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_history(history, save_path='training_history.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['accuracy'],     label='Train')
    ax1.plot(history.history['val_accuracy'], label='Val')
    ax1.set_title('Accuracy'); ax1.set_xlabel('Epoch')
    ax1.legend(); ax1.grid(True)

    ax2.plot(history.history['loss'],     label='Train')
    ax2.plot(history.history['val_loss'], label='Val')
    ax2.set_title('Loss'); ax2.set_xlabel('Epoch')
    ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f'✔ Training history saved to {save_path}')


# ── Training ──────────────────────────────────────────────────────────────────

def train(data_dir, model_save_path, epochs=50, batch_size=32, lr=0.001):
    print('=' * 60)
    print('DEEPFAKE DETECTION – MODEL TRAINING')
    print('=' * 60)

    train_dir = os.path.join(data_dir, 'train')
    val_dir   = os.path.join(data_dir, 'validation')

    if not os.path.exists(train_dir):
        raise FileNotFoundError(
            f'Training data not found at {train_dir}. '
            'Run data_preprocessing.py first.'
        )

    # ── Data generators ──────────────────────────────────────────────────────
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
    )
    val_gen = ImageDataGenerator(rescale=1./255)

    train_flow = train_gen.flow_from_directory(
        train_dir, target_size=(224, 224),
        batch_size=batch_size, class_mode='binary', shuffle=True,
    )
    val_flow = val_gen.flow_from_directory(
        val_dir, target_size=(224, 224),
        batch_size=batch_size, class_mode='binary', shuffle=False,
    )

    print(f'\nTrain samples : {train_flow.samples}')
    print(f'Val   samples : {val_flow.samples}')
    print(f'Classes       : {train_flow.class_indices}')

    # ── Class weights (handle imbalance) ─────────────────────────────────────
    classes     = np.unique(train_flow.classes)
    weights_arr = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=train_flow.classes,
    )
    class_weight = dict(zip(classes.tolist(), weights_arr.tolist()))
    print(f'Class weights : {class_weight}')

    # ── Phase 1: Train head only ──────────────────────────────────────────────
    print('\n── Phase 1: training classifier head ──')
    model, base = create_model()
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')],
    )
    model.summary()

    callbacks = [
        ModelCheckpoint(model_save_path, monitor='val_accuracy',
                        save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_loss', patience=8,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=4, min_lr=1e-7, verbose=1),
    ]

    history1 = model.fit(
        train_flow, epochs=max(epochs // 2, 10),
        validation_data=val_flow,
        class_weight=class_weight,
        callbacks=callbacks, verbose=1,
    )

    # ── Phase 2: Fine-tune top layers of backbone ─────────────────────────────
    print('\n── Phase 2: fine-tuning backbone ──')
    base.trainable = True
    # Only unfreeze the top 30 layers
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(lr / 10),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')],
    )

    history2 = model.fit(
        train_flow, epochs=epochs,
        validation_data=val_flow,
        class_weight=class_weight,
        callbacks=callbacks, verbose=1,
    )

    # Combine histories for plotting
    combined = {k: history1.history[k] + history2.history[k]
                for k in history1.history}

    class _FakeHistory:
        history = combined

    plot_history(_FakeHistory(), 'training_history.png')

    # Final evaluation
    val_loss, val_acc, val_prec, val_rec = model.evaluate(val_flow, verbose=0)
    print('\n' + '=' * 60)
    print('TRAINING COMPLETE')
    print('=' * 60)
    print(f'Val Accuracy  : {val_acc*100:.2f}%')
    print(f'Val Precision : {val_prec*100:.2f}%')
    print(f'Val Recall    : {val_rec*100:.2f}%')
    print(f'Val Loss      : {val_loss:.4f}')
    print(f'Model saved   : {model_save_path}')

    return model


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train deepfake detection model.')
    parser.add_argument('--data-dir',   default='../processed_dataset')
    parser.add_argument('--model-path', default='ml_models/cnn_model.h5')
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch-size', type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=0.001)
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f'ERROR: Dataset not found at {args.data_dir}')
        print('Run data_preprocessing.py first.')
        raise SystemExit(1)

    os.makedirs(os.path.dirname(args.model_path) or '.', exist_ok=True)
    train(args.data_dir, args.model_path, args.epochs, args.batch_size, args.lr)
