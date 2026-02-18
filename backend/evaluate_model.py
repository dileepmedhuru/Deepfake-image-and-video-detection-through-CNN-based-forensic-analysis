"""
Model Evaluation Script
Evaluate trained model on test set
"""

import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model_path, test_data_dir, batch_size=32):
    """
    Evaluate model on test set
    
    Args:
        model_path: Path to trained model
        test_data_dir: Directory containing test data
        batch_size: Batch size for evaluation
    """
    
    print("=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train_model.py")
        return
    
    model = load_model(model_path)
    print("✓ Model loaded successfully")
    
    # Load test data
    print(f"\nLoading test data from {test_data_dir}...")
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    print(f"Test samples: {test_generator.samples}")
    print(f"Classes: {test_generator.class_indices}")
    
    # Evaluate
    print("\nEvaluating model...")
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(
        test_generator,
        verbose=1
    )
    
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    print(f"Test Accuracy:  {test_accuracy*100:.2f}%")
    print(f"Test Precision: {test_precision*100:.2f}%")
    print(f"Test Recall:    {test_recall*100:.2f}%")
    print(f"Test Loss:      {test_loss:.4f}")
    
    # Get predictions
    print("\nGenerating predictions...")
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    true_classes = test_generator.classes
    
    # Classification report
    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    class_names = ['Real', 'Fake']
    print(classification_report(true_classes, predicted_classes, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("\n✓ Confusion matrix saved to confusion_matrix.png")
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall) if (test_precision + test_recall) > 0 else 0
    
    print("\n" + "=" * 50)
    print("ADDITIONAL METRICS")
    print("=" * 50)
    print(f"True Positives:  {tp}")
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"Specificity:     {specificity*100:.2f}%")
    print(f"F1-Score:        {f1_score*100:.2f}%")
    
    return {
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1_score': f1_score,
        'specificity': specificity,
        'confusion_matrix': cm
    }

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "ml_models/cnn_model.h5"
    TEST_DATA_DIR = "../processed_dataset/test"
    BATCH_SIZE = 32
    
    # Check if test data exists
    if not os.path.exists(TEST_DATA_DIR):
        print(f"Error: Test data not found at {TEST_DATA_DIR}")
        print("Please run data_preprocessing.py first!")
        exit(1)
    
    # Evaluate model
    results = evaluate_model(
        model_path=MODEL_PATH,
        test_data_dir=TEST_DATA_DIR,
        batch_size=BATCH_SIZE
    )
    
    print("\n✓ Evaluation complete!")