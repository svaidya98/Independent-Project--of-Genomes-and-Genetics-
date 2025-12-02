import sys
import pandas as pd
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
    # This function loads raw data, cleans it, engineers features, and scales it.
    # returns the processed Train and Test dataframes, and the encoders/scaler.
    
    # Pass all 7 required arguments to the data processing pipeline
    train_final, test_final, le_disorder, le_subclass, scaler = process_data_pipeline(
        train_data, 
        test_data, 
        target_gd, 
        target_subclass, 
        imputation_cols, 
        drop_high_na_cols, 
        scale_cols
    )
    
    print("\n" + "="*80)
    print(f"TRAIN DATA SHAPE: {train_final.shape}")
    print(f"TEST DATA SHAPE: {test_final.shape}")
    print("="*80 + "\n")

    # --- DATA EXPLORATION VISUALIZATIONS ---
    print("Generating Exploratory Data Visualizations...")
    
    # Target 1 Distribution
    plot_target_distribution(
        train_final, 
        target_col=target_gd, 
        title=f'Distribution of {target_gd}'
    )
    
    # Target 2 Distribution
    plot_target_distribution(
        train_final, 
        target_col=target_subclass, 
        title=f'Distribution of {target_subclass}'
    )
    
    # Feature Correlation Heatmap
    plot_correlation_heatmap(
        train_final.drop(columns=[target_gd, target_subclass]), 
        title="Feature Correlation Heatmap (Cleaned Data)"
    )

    # --- MODELING FOR GENETIC DISORDER (TARGET 1) ---

    X_gd = train_final.drop(columns=[target_gd, target_subclass])
    y_gd = train_final[target_gd]
    
    # Run Training Pipeline for target 1
    gd_results, gd_estimators, gd_winner_name = run_prediction_pipeline(
        X_gd, 
        y_gd, 
        le_disorder, 
        target_name=target_gd
    )
    
    # Retrieve the Winner (Added robust error handling for UnboundLocalError)
    winner_name = gd_winner_name
    best_model_gd = None # Initialize to prevent UnboundLocalError
    
    try:
        best_model_gd = gd_estimators[winner_name]
        print(f"\n>>> Best Model for {target_gd}: {winner_name}")
    except (KeyError, ValueError, TypeError) as e:
        print("="*80)
        print(f"[CRITICAL ERROR] Model retrieval failed for {target_gd}!")
        print(f"This means the modeling pipeline (gng_modeling.py) failed. Error: {e}")
        print("Exiting program.")
        sys.exit(1) # Stop execution if modeling fails
    
    # Final Test Evaluation
    # Prepare Test Data (Same features, target is None as test set is unlabeled)
    X_test_gd = test_final 
    y_test_gd = None # FIX: Test set is UNLABELED, must pass None for ground truth
    
    evaluate_on_test(
        best_model_gd, 
        X_test_gd, 
        y_test_gd, 
        le_disorder, 
        target_name=target_gd
    )

    # --- LOG FINAL RESULTS (TARGET 1) ---
    log_print(f"\n{'='*20} FINAL PREDICTIONS SAVED: {target_gd} {'='*20}", file_path=LOG_FILE)
    log_print(f"Winning Model: {gd_winner_name}", file_path=LOG_FILE)
    log_print(f"Predictions file: predictions_{target_gd.replace(' ', '_')}.csv", file_path=LOG_FILE)

    # --- MODELING FOR DISORDER SUBCLASS (TARGET 2) ---

    X_sub = train_final.drop(columns=[target_gd, target_subclass])
    y_sub = train_final[target_subclass]
    
    # Run Training Pipeline for target 2
    sub_results, sub_estimators, sub_winner_name = run_prediction_pipeline(
        X_sub, 
        y_sub, 
        le_subclass, 
        target_name=target_subclass
    )
    
    # Retrieve the Winner (Added robust error handling for UnboundLocalError)
    winner_name = sub_winner_name
    best_model_sub = None # Initialize to prevent UnboundLocalError
    
    try:
        best_model_sub = sub_estimators[winner_name]
        print(f"\n>>> Best Model for {target_subclass}: {winner_name}")
    except (KeyError, ValueError, TypeError) as e:
        print("="*80)
        print(f"[CRITICAL ERROR] Model retrieval failed for {target_subclass}!")
        print(f"This means the modeling pipeline (gng_modeling.py) failed. Error: {e}")
        print("Exiting program.")
        sys.exit(1) # Stop execution if modeling fails

    # Final Test Evaluation
    # Prepare Test Data (Same features, target is None as test set is unlabeled)
    X_test_sub = test_final 
    y_test_sub = None # FIX: Test set is UNLABELED, must pass None for ground truth
    
    evaluate_on_test(
        best_model_sub, 
        X_test_sub, 
        y_test_sub, 
        le_subclass, 
        target_name=target_subclass
    )
    
    # --- LOG FINAL RESULTS (TARGET 2) ---
    log_print(f"\n{'='*20} FINAL PREDICTIONS SAVED: {target_subclass} {'='*20}", file_path=LOG_FILE)
    log_print(f"Winning Model: {sub_winner_name}", file_path=LOG_FILE)
    log_print(f"Predictions file: predictions_{target_subclass.replace(' ', '_')}.csv", file_path=LOG_FILE)

if __name__ == '__main__':
    main()