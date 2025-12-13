import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import ADASYN
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import f1_score, classification_report, make_scorer
from scipy.stats import randint, uniform, loguniform
from gng_data_processing import process_data_pipeline, target_gd, target_subclass
import gng_vis as viz
import warnings
from sklearn.exceptions import FitFailedWarning
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

LOG_FILE = "modeling_results_2.txt"

def log_print(text, file_path=LOG_FILE):
    print(text)
    try:
        with open(file_path, "a", encoding='utf-8') as f:
            f.write(text + "\n")
    except Exception as e:
        print(f"Error writing to log file: {e}")

def report_tuning_results(random_search, model_name, X_val, y_val, results_summary):

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

    log_print(f"{f'--- MODELING PIPELINE: {target_name.upper()} ---':^80}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    num_classes = len(np.unique(y))
    
    log_print("Applying ADASYN to Training Data")
    try:
        adasyn = ADASYN(random_state=42, n_neighbors=5)
        X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)
        log_print(f"Resampling complete. New shape: {X_train_resampled.shape}")
    except ValueError as e:
        log_print(f"ADASYN failed (likely sparse neighborhood): {e}")
        log_print("Falling back to original training data (no resampling).")
        X_train_resampled, y_train_resampled = X_train, y_train
    
    X_train_final = X_train_resampled
    y_train_final = y_train_resampled

    scoring = {
        'F1_Weighted': make_scorer(f1_score, average='weighted', zero_division=0)
    }
    results_summary = {
        'XGBoost': {'Base F1': 0.0, 'Tuned F1': 0.0},
        'SVC': {'Base F1': 0.0, 'Tuned F1': 0.0},
        'CatBoost': {'Base F1': 0.0, 'Tuned F1': 0.0},
        'Stacking': {'Base F1': 'N/A', 'Tuned F1': 0.0} 
    }

    best_estimators = {}
    
    #XGBoost
    log_print("\n>>> XGBoost Pipeline")
    
    xgb_base = XGBClassifier(random_state=42, objective='multi:softmax', eval_metric='mlogloss')
    xgb_base.fit(X_train, y_train)
    y_pred_base = xgb_base.predict(X_val)
    f1_base_xgb = f1_score(y_val, y_pred_base, average='weighted', zero_division=0)
    results_summary['XGBoost']['Base F1'] = f1_base_xgb
    log_print(f"Baseline F1 Score: {f1_base_xgb:.4f}")

    log_print("Starting Hyperparameter Tuning...")
    xgb_param = {
    'n_estimators': randint[100, 200, 500, 1000],
    'learning_rate': uniform[0.01, 0.2],
    'max_depth': randint[3, 10],
    'min_child_weight': randint[1, 10],
    'gamma': uniform[0.5, 5],
    'subsample': uniform[0.6, 1.0],
    'colsample_bytree': uniform[0.6, 1.0]
    }
    rs_xgb = RandomizedSearchCV(XGBClassifier(objective='multi:softmax', eval_metric='mlogloss', random_state=42),
                                xgb_param, n_iter=20, scoring=scoring, refit='F1_Weighted', cv=3, n_jobs=-1, verbose=0)
    rs_xgb.fit(X_train, y_train)
    _, best_xgb = report_tuning_results(rs_xgb, 'XGBoost', X_val, y_val, results_summary)
    best_estimators['XGBoost'] = best_xgb

    #SVC
    log_print("\n>>> SVC Pipeline")

    svc_base = SVC(probability=True, random_state=42, class_weight='balanced')
    svc_base.fit(X_train, y_train)
    y_pred_base = svc_base.predict(X_val)
    f1_base_svc = f1_score(y_val, y_pred_base, average='weighted', zero_division=0)
    results_summary['SVC']['Base F1'] = f1_base_svc
    log_print(f"Baseline F1 Score: {f1_base_svc:.4f}")

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

    #Catboost
    log_print("\n>>> CatBoost Pipeline")

    cb_base = CatBoostClassifier(verbose=0, random_state=42, auto_class_weights='Balanced')
    cb_base.fit(X_train_final, y_train_final)
    y_pred_base = cb_base.predict(X_val)

    y_pred_base = np.ravel(y_pred_base)
    f1_base_cb = f1_score(y_val, y_pred_base, average='weighted', zero_division=0)
    results_summary['CatBoost']['Base F1'] = f1_base_cb
    log_print(f"Baseline F1 Score: {f1_base_cb:.4f}")

    log_print("Starting Hyperparameter Tuning...")
    cb_param = {
        'iterations': randint(100, 500),
        'depth': randint(4, 10),
        'learning_rate': uniform(0.01, 0.2),
        'l2_leaf_reg': randint(1, 10)
    }
    rs_cb = RandomizedSearchCV(CatBoostClassifier(verbose=0, random_state=42, auto_class_weights='Balanced'),
                               cb_param, n_iter=20, scoring=scoring, refit='F1_Weighted', cv=3, n_jobs=-1, verbose=0)
    rs_cb.fit(X_train_final, y_train_final)
    _, best_cb = report_tuning_results(rs_cb, 'CatBoost', X_val, y_val, results_summary)
    best_estimators['CatBoost'] = best_cb

    #Stacking Classifier
    log_print("\n>>> Training Stacking Classifier...")

    estimators = [
        ('xgb', best_xgb),
        ('svc', best_svc),
        ('cb', best_cb)
    ]
    
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=3,
        n_jobs=-1
    )
    
    stacking_clf.fit(X_train_final, y_train_final)
    
    y_pred_stack = stacking_clf.predict(X_val)
    f1_stack = f1_score(y_val, y_pred_stack, average='weighted', zero_division=0)
    results_summary['Stacking']['Tuned F1'] = f1_stack
    best_estimators['Stacking'] = stacking_clf
    log_print(f"Stacking Classifier Validation F1: {f1_stack:.4f}")


    #final model selection
    best_model_name = max(results_summary, key=lambda k: results_summary[k]['Tuned F1'])
    best_f1_score = results_summary[best_model_name]['Tuned F1']
    best_estimator = best_estimators[best_model_name]
    
    log_print(f"\n{'='*20} WINNER: {best_model_name} (F1: {best_f1_score:.4f}) {'='*20}")
    
    y_pred_best = best_estimator.predict(X_val)
    class_names = le_encoder.classes_

    clf_report = classification_report(y_val, y_pred_best, target_names=class_names, digits=4, zero_division=0)
    log_print(f"\nDETAILED CLASSIFICATION REPORT ({best_model_name.upper()} - {target_name})")
    log_print(clf_report)

    #result visualization
    print("Generating Visualizations...")
    viz.plot_model_comparison(results_summary, metric='Tuned F1')
    viz.plot_confusion_matrix_heatmap(y_val, y_pred_best, class_names, model_name=best_model_name)
    
    if best_model_name in ['XGBoost', 'CatBoost']:
        feature_names = X_train.columns
        viz.plot_feature_importance(best_estimator, feature_names, model_name=best_model_name)

    if hasattr(best_estimator, "predict_proba"):
        num_classes = len(class_names)
        viz.plot_multiclass_roc(best_estimator, X_val, y_val, num_classes, class_names=class_names)

    return results_summary, best_estimators, best_model_name