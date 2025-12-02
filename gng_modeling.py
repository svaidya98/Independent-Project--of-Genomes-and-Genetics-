import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, classification_report, make_scorer
from scipy.stats import randint, uniform, loguniform
from gng_data_processing import process_data_pipeline, target_gd, target_subclass
import gng_vis as viz
import warnings
from sklearn.exceptions import FitFailedWarning
from sklearn.impute import SimpleImputer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Define the log file name
LOG_FILE = "modeling_results.txt"

def log_print(text, file_path=LOG_FILE):
    """Prints text to console and appends it to a text file."""
    print(text)
    try:
        with open(file_path, "a", encoding='utf-8') as f:
            f.write(text + "\n")
    except Exception as e:
        print(f"Error writing to log file: {e}")

def report_tuning_results(random_search, model_name, X_val, y_val, results_summary):
    """
    Extracts results, updates summary, and logs details to file.
    """
    best_model = random_search.best_estimator_
    best_index = random_search.best_index_
    cv_results = random_search.cv_results_

    y_pred_tuned = best_model.predict(X_val)
    f1_tuned = f1_score(y_val, y_pred_tuned, average='weighted', zero_division=0)

    results_summary[model_name]['Tuned F1'] = f1_tuned

    log_print(f"\n--- {model_name} Tuning Results ---")
    log_print(f"Best Parameters: {random_search.best_params_}")
    log_print(f"Validation F1-Weighted (Tuned): {f1_tuned:.4f}")
    log_print(f"CV Best Score (Optimized Metric): {cv_results['mean_test_F1_Weighted'][best_index]:.4f}")
    
    return f1_tuned, best_model

def run_prediction_pipeline(X, y, le_encoder, target_name):
    """
    Runs the full prediction pipeline: splits data, runs BASELINE models, 
    runs TUNED models, performs Ensemble, compares results, and generates visualizations.
    """
    log_print(f"\n{'='*80}")
    log_print(f"{f'--- MODELING PIPELINE: {target_name.upper()} ---':^80}")
    log_print(f"{'='*80}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    imputer = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_val = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)
    
    scoring = {
        'F1_Weighted': make_scorer(f1_score, average='weighted', zero_division=0)
    }

    # Structure to hold both Base and Tuned scores
    results_summary = {
        'XGBoost': {'Base F1': 0.0, 'Tuned F1': 0.0},
        'Random Forest': {'Base F1': 0.0, 'Tuned F1': 0.0},
        'SVC': {'Base F1': 0.0, 'Tuned F1': 0.0},
        'VotingEnsemble': {'Base F1': 'N/A', 'Tuned F1': 0.0}
    }

    best_estimators = {}
    
    # ==========================================================================
    # 1. XGBOOST (Baseline + Tuning)
    # ==========================================================================
    log_print("\n>>> XGBoost Pipeline")
    
    # A. Baseline
    xgb_base = XGBClassifier(random_state=42, objective='multi:softmax', eval_metric='mlogloss')
    xgb_base.fit(X_train, y_train)
    y_pred_base = xgb_base.predict(X_val)
    f1_base_xgb = f1_score(y_val, y_pred_base, average='weighted', zero_division=0)
    results_summary['XGBoost']['Base F1'] = f1_base_xgb
    log_print(f"Baseline F1 Score: {f1_base_xgb:.4f}")

    # B. Tuning
    log_print("Starting Hyperparameter Tuning...")
    xgb_param = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.2)
    }
    rs_xgb = RandomizedSearchCV(XGBClassifier(objective='multi:softmax', eval_metric='mlogloss', random_state=42),
                                xgb_param, n_iter=20, scoring=scoring, refit='F1_Weighted', cv=3, n_jobs=-1, verbose=0)
    rs_xgb.fit(X_train, y_train)
    _, best_xgb = report_tuning_results(rs_xgb, 'XGBoost', X_val, y_val, results_summary)
    best_estimators['XGBoost'] = best_xgb

    # ==========================================================================
    # 2. RANDOM FOREST (Baseline + Tuning)
    # ==========================================================================
    log_print("\n>>> Random Forest Pipeline")

    # A. Baseline
    rf_base = RandomForestClassifier(random_state=42, class_weight='balanced')
    rf_base.fit(X_train, y_train)
    y_pred_base = rf_base.predict(X_val)
    f1_base_rf = f1_score(y_val, y_pred_base, average='weighted', zero_division=0)
    results_summary['Random Forest']['Base F1'] = f1_base_rf
    log_print(f"Baseline F1 Score: {f1_base_rf:.4f}")

    # B. Tuning
    log_print("Starting Hyperparameter Tuning...")
    rf_param = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(5, 20),
        'min_samples_split': randint(2, 10)
    }
    rs_rf = RandomizedSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'),
                               rf_param, n_iter=20, scoring=scoring, refit='F1_Weighted', cv=3, n_jobs=-1, verbose=0)
    rs_rf.fit(X_train, y_train)
    _, best_rf = report_tuning_results(rs_rf, 'Random Forest', X_val, y_val, results_summary)
    best_estimators['Random Forest'] = best_rf

    # ==========================================================================
    # 3. SVC (Baseline + Tuning)
    # ==========================================================================
    log_print("\n>>> SVC Pipeline")

    # A. Baseline
    svc_base = SVC(probability=True, random_state=42, class_weight='balanced')
    svc_base.fit(X_train, y_train)
    y_pred_base = svc_base.predict(X_val)
    f1_base_svc = f1_score(y_val, y_pred_base, average='weighted', zero_division=0)
    results_summary['SVC']['Base F1'] = f1_base_svc
    log_print(f"Baseline F1 Score: {f1_base_svc:.4f}")

    # B. Tuning
    log_print("Starting Hyperparameter Tuning...")
    svc_param = {
        'C': loguniform(1e-1, 10),
        'gamma': loguniform(1e-4, 1e-1),
        'kernel': ['rbf']
    }
    rs_svc = RandomizedSearchCV(SVC(probability=True, random_state=42, class_weight='balanced'),
                                svc_param, n_iter=10, scoring=scoring, refit='F1_Weighted', cv=3, n_jobs=-1, verbose=0)
    rs_svc.fit(X_train, y_train)
    _, best_svc = report_tuning_results(rs_svc, 'SVC', X_val, y_val, results_summary)
    best_estimators['SVC'] = best_svc

    # ==========================================================================
    # 4. VOTING CLASSIFIER (ENSEMBLE)
    # ==========================================================================
    log_print("\n>>> Training Voting Classifier (Ensemble)...")
    voting_clf = VotingClassifier(
        estimators=[('xgb', best_xgb), ('rf', best_rf), ('svc', best_svc)],
        voting='soft' 
    )
    voting_clf.fit(X_train, y_train)
    
    y_pred_voting = voting_clf.predict(X_val)
    f1_voting = f1_score(y_val, y_pred_voting, average='weighted', zero_division=0)
    results_summary['VotingEnsemble']['Tuned F1'] = f1_voting
    best_estimators['VotingEnsemble'] = voting_clf
    log_print(f"Voting Ensemble Validation F1: {f1_voting:.4f}")

    # ==========================================================================
    # 5. FINAL SELECTION & REPORTING
    # ==========================================================================
    
    # Determine winner based on Tuned F1
    best_model_name = max(results_summary, key=lambda k: results_summary[k]['Tuned F1'])
    best_f1_score = results_summary[best_model_name]['Tuned F1']
    best_estimator = best_estimators[best_model_name]
    
    log_print(f"\n{'='*20} WINNER: {best_model_name} (F1: {best_f1_score:.4f}) {'='*20}")
    
    y_pred_best = best_estimator.predict(X_val)
    class_names = le_encoder.classes_
    
    # Log Classification Report
    clf_report = classification_report(y_val, y_pred_best, target_names=class_names, digits=4, zero_division=0)
    log_print(f"\nDETAILED CLASSIFICATION REPORT ({best_model_name.upper()} - {target_name})")
    log_print(clf_report)

    # Visualizations (Displayed, not logged to text file)
    print("Generating Visualizations...")
    viz.plot_model_comparison(results_summary, metric='Tuned F1')
    viz.plot_confusion_matrix_heatmap(y_val, y_pred_best, class_names, model_name=best_model_name)
    
    if best_model_name in ['Random Forest', 'XGBoost']:
        viz.plot_feature_importance(best_estimator, X_train.columns, model_name=best_model_name)

    if hasattr(best_estimator, "predict_proba"):
        viz.plot_multiclass_roc(best_estimator, X_val, y_val, len(class_names), class_names=class_names)

    return results_summary, best_estimators, best_model_name