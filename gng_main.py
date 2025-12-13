import sys
import os
import pandas as pd
sys.path.append(os.getcwd())
from gng_data_processing import (
    process_data_pipeline, 
    target_gd, 
    target_subclass, 
    train_data, 
    test_data, 
    imputation_cols, 
    drop_high_na_cols, 
    scale_cols
)
from gng_modeling import run_prediction_pipeline, log_print, LOG_FILE
from gng_test import evaluate_on_test
from gng_vis import plot_target_distribution, plot_correlation_heatmap

def main():
    
    with open(LOG_FILE, 'w') as f:
        f.write("--- BIOINFORMATICS PROJECT RESULTS LOG ---\n\n")
    log_print(f"Log file initialized: {LOG_FILE}")

    train_final, test_final, le_disorder, le_subclass, scaler = process_data_pipeline(
        train_data, 
        test_data, 
        target_gd, 
        target_subclass, 
        imputation_cols, 
        drop_high_na_cols, 
        scale_cols
    ) 

    print("Generating Exploratory Data Visualizations...")
    
    if target_gd in train_final.columns:
        train_final['temp_label'] = le_disorder.inverse_transform(train_final[target_gd])
        plot_target_distribution(train_final, 'temp_label', title=f"Distribution of {target_gd}")
        train_final.drop(columns=['temp_label'], inplace=True)

    plot_correlation_heatmap(train_final.drop(columns=[target_gd, target_subclass], errors='ignore'))
    print("EDA Completed. Starting Modeling...\n")

    #modeling for target 1
    X_gd = train_final.drop(columns=[target_gd, target_subclass])
    y_gd = train_final[target_gd]

    gd_results, gd_estimators, gd_winner_name = run_prediction_pipeline(
        X_gd, 
        y_gd, 
        le_disorder, 
        target_name=target_gd
    )

    best_model_gd = gd_estimators[gd_winner_name]
    
    X_test_gd = test_final.drop(columns=[target_gd, target_subclass], errors='ignore')
    y_test_gd = test_final[target_gd] if target_gd in test_final.columns else None
    
    evaluate_on_test(best_model_gd, X_test_gd, y_test_gd, le_disorder, target_name=target_gd)
    
    #modeling for target 2
    X_sub = train_final.drop(columns=[target_gd, target_subclass], errors='ignore')
    y_sub = train_final[target_subclass]
    
    sub_results, sub_estimators, sub_winner_name = run_prediction_pipeline(
        X_sub, y_sub, le_subclass, target_name=target_subclass
    )
    
    best_model_sub = sub_estimators[sub_winner_name]
    
    X_test_sub = test_final.drop(columns=[target_gd, target_subclass], errors='ignore')
    y_test_sub = test_final[target_subclass] if target_subclass in test_final.columns else None

    evaluate_on_test(best_model_sub, X_test_sub, y_test_sub, le_subclass, target_name=target_subclass)
    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nAn error occurred: {e}")