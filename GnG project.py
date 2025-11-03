import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, make_scorer, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

train_data = r'C:\\Users\\siddh\\OneDrive\\Desktop\\Py Files\\Genetics and Genomes project\\train.csv'
test_data = r"C:\\Users\\siddh\\OneDrive\\Desktop\\Py Files\\Genetics and Genomes project\\test.csv"

train_df = pd.read_csv(train_data)
test_df = pd.read_csv(test_data)

#print(train_df.head())
#print(test_df.head())

#print(train_df.dtypes)
#print(test_df.dtypes)

def unique_values(df):
    for column in df.columns:
        unique_values = df[column].unique()
        print(f"Unique values in '{column}': {unique_values}")
#unique_values(train_df)
#unique_values(test_df)

#standardises all variations of null values in both datasets
def replace_null(df):
    df.replace({'N/A': np.nan, 
                'null': np.nan, 
                '-99': np.nan, 
                '-': np.nan, 
                'Not applicable': np.nan, 
                'nan': np.nan}, inplace=True)
    return df

train_df = replace_null(train_df)
test_df = replace_null(test_df)

#print(train_df.dtypes)
#print(test_df.dtypes)

#Renames numerical columns
train_df = train_df.rename(columns={'Mother\'s age': 'Mother Age',
                                    'Father\'s age': 'Father Age',
                                    'Heart Rate (rates/min': 'Heart Rate',
                                    'Respiratory Rate (breaths/min)': 'Respiratory Rate',
                                    'Blood cell count (mcL)': 'Blood cell count', 
                                    'White Blood cell count (thousand per microliter)': 'White Blood cell count'})
test_df = test_df.rename(columns={'Mother\'s age': 'Mother Age', 
                                  'Father\'s age': 'Father Age', 
                                  'Heart Rate (rates/min': 'Heart Rate', 
                                  'Respiratory Rate (breaths/min)': 'Respiratory Rate', 
                                  'Blood cell count (mcL)': 'Blood cell count', 
                                  'White Blood cell count (thousand per microliter)': 'White Blood cell count'})

#Changes datatype to float64 for all numerical columns
train_df = train_df.astype({'Patient Age': 'float64', 
                            'Mother Age': 'float64', 
                            'Father Age': 'float64', 
                            'Test 1': 'float64', 
                            'Test 2': 'float64', 
                            'Test 3': 'float64', 
                            'Test 4': 'float64', 
                            'Test 5': 'float64', 
                            'Symptom 1': 'float64', 
                            'Symptom 2': 'float64', 
                            'Symptom 3': 'float64', 
                            'Symptom 4': 'float64', 
                            'Symptom 5': 'float64', 
                            'No. of previous abortion': 'float64', 
                            'Blood cell count': 'float64', 
                            'White Blood cell count': 'float64'})
test_df = test_df.astype({'Patient Age': 'float64', 
                          'Mother Age': 'float64', 
                          'Father Age': 'float64', 
                          'Test 1': 'float64', 
                          'Test 2': 'float64', 
                          'Test 3': 'float64', 
                          'Test 4': 'float64', 
                          'Test 5': 'float64', 
                          'Symptom 1': 'float64', 
                          'Symptom 2': 'float64', 
                          'Symptom 3': 'float64', 
                          'Symptom 4': 'float64', 
                          'Symptom 5': 'float64', 
                          'No. of previous abortion': 'float64', 
                          'Blood cell count': 'float64', 
                          'White Blood cell count': 'float64'})

column_list = ['Patient Age', 
               'Mother Age', 
               'Father Age', 
               'Test 1', 
               'Test 2', 
               'Test 3', 
               'Test 4', 
               'Test 5', 
               'Symptom 1', 
               'Symptom 2', 
               'Symptom 3', 
               'Symptom 4', 
               'Symptom 5', 
               'No. of previous abortion',
               'Blood cell count',
               'White Blood cell count']

train_df = train_df.dropna(subset=['Genetic Disorder', 'Disorder Subclass'])


# def impute_values(df_1, df_2, column_list):
#     imputer = IterativeImputer(max_iter=10, random_state=0)
#     df_1_to_impute = df_1[column_list]
#     imputed_data = imputer.fit_transform(df_1_to_impute)
#     df_1_imputed = pd.DataFrame(imputed_data, columns=column_list, index = df.index)
#     df_1[column_list] = df_1_imputed
#     df_2_imputed_data = imputer.transform(df_2[column_list])
#     df_2_imputed = pd.DataFrame(df_2_imputed_data, columns=column_list, index = df_2.index)
#     df_2[column_list] = df_2_imputed
#     return df_1, df_2
# impute_values(train_df, test_df, column_list)

#Uses iterative imputer to fill in all the missing numerical values
imputer = IterativeImputer(max_iter=10, random_state=0)
train_df_to_impute = train_df[column_list]
train_imputed_data = imputer.fit_transform(train_df_to_impute)
train_df_imputed = pd.DataFrame(train_imputed_data, columns=column_list, index = train_df.index)
train_df[column_list] = train_df_imputed
test_imputed_data = imputer.transform(test_df[column_list])
test_df_imputed = pd.DataFrame(test_imputed_data, columns=column_list, index = test_df.index)
test_df[column_list] = test_df_imputed


train_object_columns = train_df.select_dtypes(include=['object']).columns
test_object_columns = test_df.select_dtypes(include=['object']).columns

train_df[train_object_columns] = train_df[train_object_columns].fillna('Missing')
test_df[test_object_columns] = test_df[test_object_columns].fillna('Missing')

drop_column_list = ['Patient First Name',
                    'Family Name',
                    'Patient Id',
                    'Father\'s name',
                    'Institute Name',
                    'Location of Institute',
                    ]

train_df = train_df.drop(drop_column_list, axis=1)
test_df = test_df.drop(drop_column_list, axis=1)

le_disorder = LabelEncoder()
le_subclass = LabelEncoder()
train_df['Genetic Disorder'] = le_disorder.fit_transform(train_df['Genetic Disorder'])
train_df['Disorder Subclass'] = le_subclass.fit_transform(train_df['Disorder Subclass'])
test_df['Genetic Disorder'] = le_disorder.transform(test_df['Genetic Disorder'])
test_df['Disorder Subclass'] = le_subclass.transform(test_df['Disorder Subclass'])

# 1. Get binary column names
binary_cols = [col for col in train_df.select_dtypes(include=['object']).columns if train_df[col].nunique() == 2]

# 2. Fit and store encoders on TRAIN data
encoder_dict = {}
for col in binary_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    encoder_dict[col] = le
    
# 3. Transform TEST data using stored encoders
for col in binary_cols:
    # Use the same encoder fitted on the train set
    test_df[col] = encoder_dict[col].transform(test_df[col])



# 1. Identify columns to OHE (multi-category objects remaining)
ohe_cols = [col for col in train_df.select_dtypes(include=['object']).columns if train_df[col].nunique() > 2]

# 2. Apply get_dummies to TRAIN and TEST separately
train_final = pd.get_dummies(train_df, columns=ohe_cols, drop_first=True, prefix=ohe_cols)
test_final = pd.get_dummies(test_df, columns=ohe_cols, drop_first=True, prefix=ohe_cols)

# 3. ALIGN the TEST set columns to the TRAIN set columns
# This prevents leakage and ensures the model sees the same number of features.
# It adds any missing columns to test_df (filling with 0s) and drops extra ones.
missing_cols = set(train_final.columns) - set(test_final.columns)
for c in missing_cols:
    test_final[c] = 0

# Ensure the column order is exactly the same
test_final= test_final[train_final.columns]

# NOTE: For simplicity, the above code assumes 'train_df' and 'test_df'
# were updated in place in step 2. You would need to use concat/merge logic
# if you didn't do it in place.


#Bar graphs
GD_count = train_final['Genetic Disorder'].value_counts()
GD_count.plot(kind = 'bar')
plt.xlabel('Genetic Disorder')
plt.ylabel('Count')
plt.title('Genetic Disorder Distribution')
plt.show()

DS_count = train_final['Disorder Subclass'].value_counts()
DS_count.plot(kind = 'bar')
plt.xlabel('Disorder Subclass')
plt.ylabel('Count')
plt.title('Disorder Subclass Distribution')
plt.show()

#Box plots - Distributions
plt.figure(figsize=(8, 6))
train_final['Patient Age'].hist(bins=20)
train_final['Mother Age'].hist(bins=20)
train_final['Father Age'].hist(bins=20)
plt.title('Patient and Parent Age Distribution')
plt.show()

plt.figure(figsize=(8, 6))
train_final['Test 1'].hist(bins=20)
train_final['Test 2'].hist(bins=20)
train_final['Test 3'].hist(bins=20)
train_final['Test 4'].hist(bins=20)
train_final['Test 5'].hist(bins=20)
plt.title('Test Scores Distribution')
plt.show()

plt.figure(figsize=(8, 6))
train_final['Blood cell count'].hist(bins=20)
train_final['White Blood cell count'].hist(bins=20)
plt.title('Blood count and White Blood count Distribution')
plt.show()

#Relation between target and features

plt.figure(figsize=(10, 6))
train_final.boxplot(column=['Test 1', 'Test 2', 'Test 3', 'Test 4', 'Test 5'], by='Genetic Disorder')
plt.title('Test by Genetic Disorder')
plt.suptitle('') # Suppress automatic title
plt.show()

plt.figure(figsize=(10, 6))
train_final.boxplot(column=['Test 1', 'Test 2', 'Test 3', 'Test 4', 'Test 5'], by='Disorder Subclass')
plt.title('Test by Genetic Disorder')
plt.suptitle('') # Suppress automatic title
plt.show()

plt.figure(figsize=(10, 6))
train_final.boxplot(column=['Patient Age', 'No. of previous abortion', 'Blood cell count', 'White Blood cell count'], by='Genetic Disorder')
plt.title('Others by Genetic Disorder')
plt.suptitle('') # Suppress automatic title
plt.show()

plt.figure(figsize=(10, 6))
train_final.boxplot(column=['Patient Age', 'No. of previous abortion', 'Blood cell count', 'White Blood cell count'], by='Disorder Subclass')
plt.title('Others by Genetic Disorder')
plt.suptitle('') # Suppress automatic title
plt.show()

plt.figure(figsize=(10, 6))
train_final.boxplot(column=['Symtom 1', 'Symtom 2', 'Symtom 3', 'Symtom 4', 'Symtom 5'], by='Disorder Subclass')
plt.title('Symptoms by Genetic Disorder')
plt.suptitle('') # Suppress automatic title
plt.show()

plt.figure(figsize=(10, 6))
train_final.boxplot(column=['Symtom 1', 'Symtom 2', 'Symtom 3', 'Symtom 4', 'Symtom 5'], by='Genetic Disorder')
plt.title('Symptoms by Genetic Disorder')
plt.suptitle('') # Suppress automatic title
plt.show()

#HeatMap
numeric_cols = train_final.select_dtypes(include=np.number).columns
corr_matrix = train_final[numeric_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features')
plt.show()


#Feature engineering
train_final['Parent_Age_Diff'] = (train_final['Mother Age'] - train_final['Father Age']).abs()
test_final['Parent_Age_Diff'] = (test_final['Mother Age'] - test_final['Father Age']).abs()

test_cols = [f'Test {i}' for i in range(1, 6)]
train_final['Health_Test_Mean'] = train_final[test_cols].mean(axis=1)
test_final['Health_Test_Mean'] = test_final[test_cols].mean(axis=1)

symptom_cols = [f'Symptom {i}' for i in range(1, 6)]
train_final['Total_Symptom_Count'] = train_final[symptom_cols].sum(axis=1)
test_final['Total_Symptom_Count'] = test_final[symptom_cols].sum(axis=1)

#Normalization
scale_cols = ['Patient Age', 'Mother Age', 'Father Age',
              'Test 1', 'Test 2', 'Test 3', 'Test 4', 'Test 5',
              'Blood cell count', 'White Blood cell count',
              'No. of previous abortion', 'Parent_Age_Diff', 'Health_Test_Mean']

scaler = StandardScaler()
train_final[scale_cols] = scaler.fit_transform(train_final[scale_cols])
test_final[scale_cols] = scaler.transform(test_final[scale_cols])


#Training split
X = train_final.drop('Genetic Disorder', axis=1)
y = train_final['Genetic Disorder']

# Split the data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

#Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Initial Random Forest Accuracy: {accuracy:.2f}")

#RF Random Search CV
model = RandomForestClassifier(random_state=42)
param_distributions = {
    'n_estimators': randint(low=10, high=200),
    'max_depth': randint(low=1, high=20),
    'criterion': ['gini', 'entropy']
}

# 'estimator': The model to tune.
# 'param_distributions': The dictionary of hyperparameter distributions.
# 'n_iter': The number of random combinations to try.
# 'cv': Number of cross-validation folds.
# 'scoring': The metric to optimize (e.g., 'accuracy' for classification).
# 'random_state': For reproducibility of random sampling.

scoring = {
    'F1_Weighted': 'f1_weighted',    # <-- Metric to optimize for class imbalance
    'Accuracy': 'accuracy',          # <-- Standard measure
    'Recall_Macro': 'recall_macro',  # <-- Useful for ensuring we don't miss positive cases
    'Precision_Macro': 'precision_macro'
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=100,  
    cv=5,       
    scoring=scoring,
    refit='F1_Weighted', 
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train, y_train)
print("Best Parameters found:", random_search.best_params_)
print("Best Cross-Validation Score:", random_search.best_score_)

best_f1_model = random_search.best_estimator_
y_val_tuned_pred = best_f1_model.predict(X_val)
tuned_accuracy = accuracy_score(y_val, y_val_tuned_pred)
print(f"Tuned Random Forest Accuracy: {tuned_accuracy:.2f}")


#XGBoost Classifier
xgb_model = XGBClassifier(
    objective='multi:softmax',  # For multi-class classification
    num_class=len(set(y)),      # Number of unique classes in the target variable
    n_estimators=100,           # Number of boosting rounds (trees)
    learning_rate=0.1,          # Step size shrinkage to prevent overfitting
    max_depth=3,                # Maximum depth of a tree
    use_label_encoder=False,    # Suppress warning about label encoder deprecation
    eval_metric='mlogloss'      # Evaluation metric for multi-class classification
)

# 4. Train the model
xgb_model.fit(X_train, y_train)

# 5. Make predictions on the test set
y_pred = xgb_model.predict(X_val)

# 6. Evaluate the model's performance
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Optional: Print classification report and confusion matrix for detailed evaluation
print("\nClassification Report:")
print(classification_report(y_val, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))

#XGBoost Hyperparameter tuning
xgb_tune = xgb_model.XGBClassifier(objective='binary:logistic', eval_metric='logloss')

param_distributions = {
    'n_estimators': randint(50, 500),
    'learning_rate': uniform(0.01, 0.3),
    'max_depth': randint(3, 10),
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': uniform(0.5, 1.0),
    'gamma': uniform(0, 0.5),
    'reg_alpha': [0, 0.001, 0.01, 0.1]
}

random_search = RandomizedSearchCV(
    estimator=xgb_tune,
    param_distributions=param_distributions,
    n_iter=100,
    scoring=scoring,
    refit='F1_Weighted',
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
print(f"Best parameters: {random_search.best_params_}")
print(f"Best F1 weighted score: {random_search.best_score_}")
best_xgb_model = random_search.best_estimator_
y_val_tuned_pred = best_xgb_model.predict(X_val)
tuned_accuracy = f1_score(y_val, y_val_tuned_pred)
print(f"Tuned XGBoost F1 score: {tuned_accuracy:.2f}")









