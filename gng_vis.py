import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

#Data Exploration Plots
def plot_target_distribution(df, target_col, title="Target Distribution"):

    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x=target_col, data=df, palette='viridis')
    plt.title(title)
    plt.xlabel('Target Class')
    plt.ylabel('Count')
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    plt.show()

def plot_correlation_heatmap(df, title="Correlation Heatmap"):
    numeric_df = df.select_dtypes(include=[np.number])
    
    plt.figure(figsize=(14, 12))
    corr_matrix = numeric_df.corr()

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', vmax=.8, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=False)
    plt.title(title)
    plt.show()

def plot_feature_distribution(df, feature, target=None):

    plt.figure(figsize=(10, 6))
    if target:
        sns.boxplot(x=target, y=feature, data=df, palette='Set2')
        plt.title(f'Distribution of {feature} by {target}')
    else:
        sns.histplot(df[feature], kde=True, color='skyblue')
        plt.title(f'Distribution of {feature}')
    plt.show()

#Result Plots
def plot_model_comparison(results_summary, metric='Tuned F1'):

    models = list(results_summary.keys())
    scores = [results_summary[m][metric] for m in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    plt.ylim(0, 1.1)
    plt.title(f'Model Comparison: {metric}')
    plt.ylabel('Score')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}', ha='center', va='bottom')
    plt.show()

def plot_confusion_matrix_heatmap(y_true, y_pred, class_names, model_name="Model"):

    cm = confusion_matrix(y_true, y_pred)

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Normalized Confusion Matrix: {model_name}')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.show()

def plot_feature_importance(model, feature_names, top_n=15, model_name="Model"):

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        print(f"Model {model_name} does not expose feature_importances_")
        return

    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    feat_imp.plot(kind='barh', color='teal')
    plt.title(f'Top {top_n} Feature Importances ({model_name})')
    plt.xlabel('Importance Score')
    plt.gca().invert_yaxis()
    plt.show()

def plot_multiclass_roc(model, X_val, y_val, n_classes, class_names=None):

    y_val_bin = label_binarize(y_val, classes=np.arange(n_classes))
    y_score = model.predict_proba(X_val)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    plt.figure(figsize=(10, 8))
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_val_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        label = f'Class {i}'
        if class_names is not None:
            label = f'{class_names[i]}'
            
        plt.plot(fpr[i], tpr[i], lw=2, label=f'{label} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class ROC Curves (One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.show()

def plot_pr_curve_comparison(model_dict, X_val, y_val):

    n_classes = len(np.unique(y_val))
    y_val_bin = label_binarize(y_val, classes=np.arange(n_classes))
    
    plt.figure(figsize=(10, 8))
    
    for name, model in model_dict.items():
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_val)
            precision, recall, _ = precision_recall_curve(y_val_bin.ravel(), y_score.ravel())
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, lw=2, label=f'{name} (Micro-AUC = {pr_auc:.3f})')
            
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Averaged Precision-Recall Curve Comparison')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.show()
