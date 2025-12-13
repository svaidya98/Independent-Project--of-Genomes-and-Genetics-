import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import ADASYN

train_data = r'C:\\Users\\siddh\\OneDrive\\Desktop\\Py Files\\Genetics and Genomes project\\train.csv'
test_data = r'C:\\Users\\siddh\\OneDrive\\Desktop\\Py Files\\Genetics and Genomes project\\test.csv'


target_gd = 'Genetic Disorder'
target_subclass = 'Disorder Subclass'

#columns for numerical imputation
imputation_cols = [
    'Patient Age', 'Mother Age', 'Father Age', 'Test 1', 'Test 2', 'Test 3', 
    'Test 4', 'Test 5', 'Blood cell count', 'White Blood cell count',
    'No. of previous abortion'
]

#columns to drop
drop_high_na_cols = [
    'Institute Name', 'Location of Institute', 'Patient First Name', 
    'Family Name', 'Father\'s name', 'Maternal gene', 'Paternal gene', 
    'Folic acid details (peri-conceptional)', 'H/O serious maternal illness', 
    'H/O radiation exposure (x-ray)', 'H/O substance abuse', 
    'History of anomalies in previous pregnancies'
]

#columns to be scaled
scale_cols = [
    'Patient Age', 'Mother Age', 'Father Age', 'Test 1', 'Test 2', 'Test 3', 
    'Test 4', 'Test 5', 'Blood cell count', 'White Blood cell count',
    'No. of previous abortion', 'Parent_Age_Diff', 'Health_Test_Mean',
    'Total_Symptom_Count', 'WBC_to_BloodCell_Ratio', 'Parental_Risk_Count',
    'Pregnancy_Risk_Index_Mult'
]

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please check the path.")
        return pd.DataFrame()

def replace_null(df):
    df.replace({'N/A': np.nan, 
                'null': np.nan, 
                '-99': np.nan, 
                '-': np.nan, 
                'Not applicable': np.nan, 
                'nan': np.nan}, inplace=True)
    return df

def rename_columns(df):
    rename_map = {
        'Mother\'s age': 'Mother Age',
        'Father\'s age': 'Father Age',
        'Heart Rate (rates/min': 'Heart Rate',
        'Respiratory Rate (breaths/min)': 'Respiratory Rate',
        'Blood cell count (mcL)': 'Blood cell count', 
        'White Blood cell count (thousand per microliter)': 'White Blood cell count',
        'Patient Age': 'Patient Age',
        'Test 1': 'Test 1',
        'Test 2': 'Test 2',
        'Test 3': 'Test 3',
        'Test 4': 'Test 4',
        'Test 5': 'Test 5',
        'No. of previous abortion': 'No. of previous abortion',
        'History of anomalies in previous pregnancies': 'History of anomalies in previous pregnancies',
        'Genes in mother\'s side': 'Genes in mother\'s side',
        'Inherited from father': 'Genes in father\'s side',
        'Maternal gene': 'Maternal gene',
        'Paternal gene': 'Paternal gene'
    }
    df.rename(columns=rename_map, inplace=True)
    return df

def encode_targets(train_df, test_df, target_gd, target_subclass):
    le_disorder = LabelEncoder()
    le_subclass = LabelEncoder()

    train_df[target_gd] = le_disorder.fit_transform(train_df[target_gd].astype(str))
    train_df[target_subclass] = le_subclass.fit_transform(train_df[target_subclass].astype(str))

    if target_gd in test_df.columns:
        test_df[target_gd] = le_disorder.transform(test_df[target_gd].astype(str))
    if target_subclass in test_df.columns:
        test_df[target_subclass] = le_subclass.transform(test_df[target_subclass].astype(str))
    
    return train_df, test_df, le_disorder, le_subclass

def convert_to_float(df):
    cols_to_convert = ['Patient Age', 'Mother Age', 'Father Age', 'Blood cell count', 
                       'White Blood cell count', 'Test 1', 'Test 2', 'Test 3', 
                       'Test 4', 'Test 5', 'No. of previous abortion']
    for col in cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def impute_values(train_df, test_df, imputation_cols):
    imputer = IterativeImputer(random_state=42)
    imputer.fit(train_df[imputation_cols])

    train_df[imputation_cols] = imputer.transform(train_df[imputation_cols])
    test_df[imputation_cols] = imputer.transform(test_df[imputation_cols])
    
    return train_df, test_df

def feature_engineer(train_df, test_df):
    bool_cols = ['Genes in mother\'s side', 'Genes in father\'s side', 'Maternal gene', 
                 'Paternal gene', 'Parental consent', 'Birth asphyxia', 
                 'Assisted conception IVF/ART', 'Symptom 1', 'Symptom 2', 
                 'Symptom 3', 'Symptom 4', 'Symptom 5']
    for col in bool_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].map({'Yes': 1, 'No': 0, 1.0: 1, 0.0: 0})
        if col in test_df.columns:
            test_df[col] = test_df[col].map({'Yes': 1, 'No': 0, 1.0: 1, 0.0: 0})

    train_df['Parent_Age_Diff'] = abs(train_df['Mother Age'] - train_df['Father Age'])
    test_df['Parent_Age_Diff'] = abs(test_df['Mother Age'] - test_df['Father Age'])

    test_cols = [f'Test {i}' for i in range(1, 6)]
    train_df['Health_Test_Mean'] = train_df[test_cols].mean(axis=1)
    test_df['Health_Test_Mean'] = test_df[test_cols].mean(axis=1)

    symptom_cols = [f'Symptom {i}' for i in range(1, 6)]
    train_df['Total_Symptom_Count'] = train_df[symptom_cols].sum(axis=1)
    test_df['Total_Symptom_Count'] = test_df[symptom_cols].sum(axis=1)

    train_df['WBC_to_BloodCell_Ratio'] = train_df['White Blood cell count'] / train_df['Blood cell count']
    test_df['WBC_to_BloodCell_Ratio'] = test_df['White Blood cell count'] / test_df['Blood cell count']

    risk_cols = ['Genes in mother\'s side', 'Genes in father\'s side', 'Maternal gene', 'Paternal gene']
    train_df['Parental_Risk_Count'] = train_df[risk_cols].sum(axis=1)
    test_df['Parental_Risk_Count'] = test_df[risk_cols].sum(axis=1)
    
    if 'History of anomalies in previous pregnancies' in train_df.columns:
        train_df['Pregnancy_Risk_Index_Mult'] = train_df['No. of previous abortion'] * train_df['History of anomalies in previous pregnancies'].map({'Yes': 1, 'No': 0, 1:1, 0:0})
        test_df['Pregnancy_Risk_Index_Mult'] = test_df['No. of previous abortion'] * test_df['History of anomalies in previous pregnancies'].map({'Yes': 1, 'No': 0, 1:1, 0:0})
    
    return train_df, test_df

def apply_resampling(X, y):

    print("Applying ADASYN")
    try:
        adasyn = ADASYN(random_state=42, sampling_strategy='auto')
        X_resampled, y_resampled = adasyn.fit_resample(X, y)
        print(f"Original shape: {X.shape}, Resampled shape: {X_resampled.shape}")
        return X_resampled, y_resampled
    except ValueError as e:
        print(f"ADASYN failed (likely due to extremely rare class): {e}")
        print("Falling back to original data.")
        return X, y

def drop_and_ohe(df, drop_cols, target_gd=None, target_subclass=None):
    df_temp = df.copy()

    cols_to_drop = [c for c in ['Patient Id'] + drop_cols if c in df_temp.columns]
    df_temp.drop(columns=cols_to_drop, errors='ignore', inplace=True)

    targets = [t for t in [target_gd, target_subclass] if t is not None]
    
    categorical_cols = df_temp.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c not in targets]

    force_ohe = ['Status', 'Respiratory Rate', 'Heart Rate', 'Follow-up', 'Gender', 
                 'Autopsy shows birth defect', 'Place of birth', 'Birth defects', 'Blood test result']
    categorical_cols = list(set(categorical_cols + [c for c in force_ohe if c in df_temp.columns]))

    df_temp = pd.get_dummies(df_temp, columns=categorical_cols, dummy_na=False, drop_first=True)
    
    return df_temp

def scale_features(train_df, test_df, columns_to_scale):

    scaler = StandardScaler()
    train_df[columns_to_scale] = scaler.fit_transform(train_df[columns_to_scale])
    test_df[columns_to_scale] = scaler.transform(test_df[columns_to_scale])
    return train_df, test_df, scaler

def process_data_pipeline(train_data, test_data, target_gd, target_subclass, imputation_cols, drop_high_na_cols, scale_cols):
    print("Starting data processing pipeline...")
    train_df = load_data(train_data)
    test_df = load_data(test_data)
    
    train_df = replace_null(train_df)
    test_df = replace_null(test_df)
    train_df = rename_columns(train_df)
    test_df = rename_columns(test_df)
    
    train_df, test_df, le_disorder, le_subclass = encode_targets(train_df, test_df, target_gd, target_subclass)
    
    train_df = convert_to_float(train_df)
    test_df = convert_to_float(test_df)
    train_df, test_df = impute_values(train_df, test_df, imputation_cols)
    train_df, test_df = feature_engineer(train_df, test_df)

    train_final = drop_and_ohe(train_df, drop_high_na_cols, target_gd, target_subclass)
    test_final = drop_and_ohe(test_df, drop_high_na_cols)

    train_cols = train_final.drop(columns=[target_gd, target_subclass], errors='ignore').columns
    test_final = test_final.reindex(columns=train_cols, fill_value=0)

    numeric_cols = train_final.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in [target_gd, target_subclass]]

    train_final[numeric_cols] = train_final[numeric_cols].fillna(0)
    test_final[numeric_cols] = test_final[numeric_cols].fillna(0)

    train_final, test_final, scaler = scale_features(train_final, test_final, scale_cols)
    
    print("Data processing complete.")
    return train_final, test_final, le_disorder, le_subclass, scaler