import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import gng_vis as viz
from gng_modeling import log_print

def evaluate_on_test(model, X_test, y_test, le_encoder, target_name):

    print(f"{f'FINAL TEST EVALUATION: {target_name.upper()}':^80}")

    y_pred = model.predict(X_test)
    y_pred_labels = le_encoder.inverse_transform(y_pred)

    if y_test is None or len(np.unique(y_test)) < 2:
        print(f"\n[INFO] Test data appears UNLABELED. Skipping metrics.")
        print(f"First 10 Predictions for {target_name}:")
        print(y_pred_labels[:10])
        
        #save to csv
        output_filename = f"predictions_{target_name.replace(' ', '_')}.csv"
        pd.DataFrame(y_pred_labels, columns=[target_name]).to_csv(output_filename, index=False)
        print(f"Predictions saved to {output_filename}")
        return
    
    final_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    print(f"Test Set F1-Weighted Score: {final_f1:.4f}")

    class_names = le_encoder.classes_
    print("Classification Report (Test Set)")
    report_str = classification_report(y_test, y_pred, target_names=class_names, digits=4, zero_division=0)
    log_print(report_str)
    
    print("Generating Test Set Visualizations...")

    try:
        #convert numeric y_test back to string labels
        y_test_labels = le_encoder.inverse_transform(y_test)
        
        #explicitly pass 'labels' to ensure matrix shape matches class_names
        cm = confusion_matrix(y_test_labels, y_pred_labels, labels=class_names)

        with np.errstate(divide='ignore', invalid='ignore'):
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f"Normalized Confusion Matrix: Best Model ({target_name})")
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.show()
    except Exception as e:
        print(f"Could not generate Confusion Matrix: {e}")

    if hasattr(model, "predict_proba"):
        num_classes = len(class_names)
        viz.plot_multiclass_roc(
            model, 
            X_test, 
            y_test, 
            num_classes, 
            class_names=class_names
        )
