import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score
import gng_vis as viz

def evaluate_on_test(model, X_test, y_test, le_encoder, target_name):
    """
    Applies the model to the Test Set. 
    Handles unlabeled data gracefully.
    """
    print("\n" + "="*80)
    print(f"{f'FINAL TEST EVALUATION: {target_name.upper()}':^80}")
    print("="*80)

    # 1. Generate Predictions
    print("Generating predictions on Test Set...")
    y_pred = model.predict(X_test)
    
    # Convert numeric predictions back to string labels
    y_pred_labels = le_encoder.inverse_transform(y_pred)
    
    # --- CHECK IF GROUND TRUTH EXISTS AND IS VALID ---
    # If y_test is None or contains only one unique value (often a placeholder like -1), skip metrics
    if y_test is None or len(np.unique(y_test)) < 2:
        print(f"\n[INFO] Test data appears UNLABELED (no ground truth). Skipping metrics.")
        print(f"First 10 Predictions for {target_name}:")
        print(y_pred_labels[:10])
        
        # Save to CSV
        output_filename = f"predictions_{target_name.replace(' ', '_')}.csv"
        pd.DataFrame(y_pred_labels, columns=[target_name]).to_csv(output_filename, index=False)
        print(f"Predictions saved to {output_filename}")
        return

    # --- IF LABELS EXIST, PROCEED WITH EVALUATION ---
    
    final_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    print(f"Test Set F1-Weighted Score: {final_f1:.4f}")

    class_names = le_encoder.classes_
    print("\n" + "-"*50)
    print("Classification Report (Test Set)")
    print("-" * 50)
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4, zero_division=0))
    
    print("Generating Test Set Visualizations...")
    viz.plot_confusion_matrix_heatmap(
        y_test, 
        y_pred, 
        class_names, 
        model_name=f"Best Model ({target_name})"
    )

    if hasattr(model, "predict_proba"):
        num_classes = len(class_names)
        viz.plot_multiclass_roc(
            model, 
            X_test, 
            y_test, 
            num_classes, 
            class_names=class_names
        )
