# evaluate_model.py
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

def evaluate_model(model_path, test_dir, save_plots=True):
    """
    Comprehensive model evaluation
    
    Args:
        model_path: Path to trained model
        test_dir: Path to test dataset directory
        save_plots: Whether to save visualization plots
    """
    print("="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = keras.models.load_model(model_path)
    print("✓ Model loaded successfully")
    
    # Prepare test data
    print(f"\nLoading test data from: {test_dir}")
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"✓ Found {test_generator.samples} test images")
    print(f"  Classes: {list(test_generator.class_indices.keys())}")
    
    # Get predictions
    print("\nMaking predictions...")
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # Get class names
    class_names = list(test_generator.class_indices.keys())
    
    # Overall accuracy
    accuracy = np.mean(y_pred == y_true)
    print(f"\n{'='*70}")
    print(f"OVERALL ACCURACY: {accuracy*100:.2f}%")
    print(f"{'='*70}")
    
    # Classification report
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    report = classification_report(
        y_true, y_pred, 
        target_names=class_names,
        digits=4
    )
    print(report)
    
    # Save classification report
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True
    )
    
    # Confusion matrix
    print("\n" + "="*70)
    print("CONFUSION MATRIX")
    print("="*70)
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Per-class metrics
    print("\n" + "="*70)
    print("PER-CLASS METRICS")
    print("="*70)
    print(f"{'Class':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-"*70)
    
    for i, class_name in enumerate(class_names):
        class_accuracy = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        precision = report_dict[class_name]['precision']
        recall = report_dict[class_name]['recall']
        f1 = report_dict[class_name]['f1-score']
        
        print(f"{class_name:<15} {class_accuracy*100:>8.2f}%  {precision:>8.4f}  {recall:>8.4f}  {f1:>8.4f}")
    
    # Calculate AUC for each class
    print("\n" + "="*70)
    print("AUC SCORES (One-vs-Rest)")
    print("="*70)
    
    y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=len(class_names))
    
    for i, class_name in enumerate(class_names):
        try:
            auc = roc_auc_score(y_true_onehot[:, i], predictions[:, i])
            print(f"{class_name:<15}: {auc:.4f}")
        except:
            print(f"{class_name:<15}: N/A")
    
    # Misclassification analysis
    print("\n" + "="*70)
    print("MISCLASSIFICATION ANALYSIS")
    print("="*70)
    
    misclassified = y_pred != y_true
    num_misclassified = np.sum(misclassified)
    
    print(f"Total misclassifications: {num_misclassified} ({num_misclassified/len(y_true)*100:.2f}%)")
    print("\nMost common misclassifications:")
    
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                print(f"  {class_names[i]} → {class_names[j]}: {cm[i, j]} times")
    
    if save_plots:
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        
        # 1. Confusion Matrix Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: confusion_matrix.png")
        plt.close()
        
        # 2. Normalized Confusion Matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2%',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Percentage'}
        )
        plt.title('Normalized Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: confusion_matrix_normalized.png")
        plt.close()
        
        # 3. Per-class metrics bar chart
        metrics = ['precision', 'recall', 'f1-score']
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, metric in enumerate(metrics):
            values = [report_dict[cls][metric] for cls in class_names]
            axes[idx].bar(class_names, values, color='skyblue', edgecolor='navy')
            axes[idx].set_title(metric.replace('-', ' ').title(), fontsize=12, fontweight='bold')
            axes[idx].set_ylim([0, 1])
            axes[idx].set_ylabel('Score')
            axes[idx].grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('per_class_metrics.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: per_class_metrics.png")
        plt.close()
        
        # 4. Prediction confidence distribution
        plt.figure(figsize=(12, 6))
        confidence_scores = np.max(predictions, axis=1)
        correct_predictions = y_pred == y_true
        
        plt.hist(confidence_scores[correct_predictions], bins=20, alpha=0.7, 
                label='Correct Predictions', color='green', edgecolor='black')
        plt.hist(confidence_scores[~correct_predictions], bins=20, alpha=0.7,
                label='Incorrect Predictions', color='red', edgecolor='black')
        plt.xlabel('Confidence Score', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('confidence_distribution.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: confidence_distribution.png")
        plt.close()
    
    # Save evaluation results to JSON
    evaluation_results = {
        'timestamp': datetime.now().isoformat(),
        'model_path': model_path,
        'test_samples': int(test_generator.samples),
        'overall_accuracy': float(accuracy),
        'per_class_metrics': {
            cls: {
                'precision': float(report_dict[cls]['precision']),
                'recall': float(report_dict[cls]['recall']),
                'f1_score': float(report_dict[cls]['f1-score']),
                'support': int(report_dict[cls]['support'])
            }
            for cls in class_names
        },
        'confusion_matrix': cm.tolist(),
        'macro_avg': {
            'precision': float(report_dict['macro avg']['precision']),
            'recall': float(report_dict['macro avg']['recall']),
            'f1_score': float(report_dict['macro avg']['f1-score'])
        },
        'weighted_avg': {
            'precision': float(report_dict['weighted avg']['precision']),
            'recall': float(report_dict['weighted avg']['recall']),
            'f1_score': float(report_dict['weighted avg']['f1-score'])
        }
    }
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print("\n✓ Saved: evaluation_results.json")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    
    return evaluation_results

if __name__ == "__main__":
    import sys
    
    # Default paths
    model_path = 'wound_classifier_final.h5'
    test_dir = 'wound_dataset/test'
    
    # Check if custom paths provided
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    if len(sys.argv) > 2:
        test_dir = sys.argv[2]
    
    # Run evaluation
    try:
        evaluate_model(model_path, test_dir, save_plots=True)
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        print("\nUsage: python evaluate_model.py [model_path] [test_dir]")
        print("Example: python evaluate_model.py wound_classifier_final.h5 wound_dataset/test")
        sys.exit(1)
