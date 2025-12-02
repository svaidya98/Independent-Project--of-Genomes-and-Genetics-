import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize


# --- Configuration Constants ---
train_data = r'C:\\Users\\siddh\\OneDrive\\Desktop\\Py Files\\Genetics and Genomes project\\train.csv'
test_data = r'C:\\Users\\siddh\\OneDrive\\Desktop\\Py Files\\Genetics and Genomes project\\test.csv'


target_gd = 'Genetic Disorder'
target_subclass = 'Disorder Subclass'

# Columns for numerical imputation (IterativeImputer)
imputation_cols = [
    'Patient Age', 'Mother Age', 'Father Age', 'Test 1', 'Test 2', 'Test 3', 
    'Test 4', 'Test 5', 'Blood cell count', 'White Blood cell count',
    'No. of previous abortion'
]

# Columns to drop (high missing values, identifiers, non-predictive text)
drop_high_na_cols = [
    'Institute Name', 'Location of Institute', 'Patient First Name', 
    'Family Name', 'Father\'s name', 'Maternal gene', 'Paternal gene', 
    'Folic acid details (peri-conceptional)', 'H/O serious maternal illness', 
    'H/O radiation exposure (x-ray)', 'H/O substance abuse', 
    'History of anomalies in previous pregnancies'
]

# Columns to be scaled after feature engineering and imputation
scale_cols = [
    'Patient Age', 'Mother Age', 'Father Age', 'Test 1', 'Test 2', 'Test 3', 
    'Test 4', 'Test 5', 'Blood cell count', 'White Blood cell count',
    'No. of previous abortion', 'Parent_Age_Diff', 'Health_Test_Mean',
    'Total_Symptom_Count', 'WBC_to_BloodCell_Ratio', 'Parental_Risk_Count',
    'Pregnancy_Risk_Index_Mult'
]

# --- Helper Functions (From previous snippets) ---

def load_data(file_path):
    """Loads data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please check the path.")
        return pd.DataFrame()

def replace_null(df):
    """Replaces various null representations with numpy NaN."""
    df.replace({'N/A': np.nan, 
                'null': np.nan, 
                '-99': np.nan, 
                '-': np.nan, 
                'Not applicable': np.nan, 
                'nan': np.nan}, inplace=True)
    return df

def rename_columns(df):
    """Renames inconsistent or special-character columns."""
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
        'Inherited from father': 'Genes in father\'s side', # Renaming for consistency
        'Maternal gene': 'Maternal gene',
        'Paternal gene': 'Paternal gene'
    }
    df.rename(columns=rename_map, inplace=True)
    return df

def encode_targets(train_df, test_df, target_gd, target_subclass):
    """Encodes the target variables using LabelEncoder."""
    le_disorder = LabelEncoder()
    le_subclass = LabelEncoder()

    # Fit and transform on training data
    train_df[target_gd] = le_disorder.fit_transform(train_df[target_gd].astype(str))
    train_df[target_subclass] = le_subclass.fit_transform(train_df[target_subclass].astype(str))
    
    # Transform test data (targets are dropped from X later, but necessary for full data cleaning)
    # The test set does not have the 'Disorder Subclass' column, so we handle it.
    if target_gd in test_df.columns:
        test_df[target_gd] = le_disorder.transform(test_df[target_gd].astype(str))
    if target_subclass in test_df.columns:
        # Note: If the test set has the subclass, transform it. Otherwise, this is skipped.
        # This is a safety measure; typically, the true test set target is unknown.
        test_df[target_subclass] = le_subclass.transform(test_df[target_subclass].astype(str))
    
    return train_df, test_df, le_disorder, le_subclass

def convert_to_float(df):
    """Converts specific columns to float type."""
    cols_to_convert = ['Patient Age', 'Mother Age', 'Father Age', 'Blood cell count', 
                       'White Blood cell count', 'Test 1', 'Test 2', 'Test 3', 
                       'Test 4', 'Test 5', 'No. of previous abortion']
    for col in cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def impute_values(train_df, test_df, imputation_cols):
    """Imputes missing numerical values using IterativeImputer."""
    imputer = IterativeImputer(random_state=42)
    
    # Fit imputer on the training data only
    imputer.fit(train_df[imputation_cols])
    
    # Transform both train and test data
    train_df[imputation_cols] = imputer.transform(train_df[imputation_cols])
    test_df[imputation_cols] = imputer.transform(test_df[imputation_cols])
    
    return train_df, test_df

def feature_engineer(train_df, test_df):
    """Creates new, potentially predictive features."""
    
    # Standardize Yes/No/Boolean columns to numeric (1/0)
    bool_cols = [
        'Genes in mother\'s side', 'Genes in father\'s side', 'Maternal gene', 
        'Paternal gene', 'Parental consent', 'Birth asphyxia', 
        'Assisted conception IVF/ART', 'Symptom 1', 'Symptom 2', 
        'Symptom 3', 'Symptom 4', 'Symptom 5'
    ]
    for col in bool_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].map({'Yes': 1, 'No': 0, 1.0: 1, 0.0: 0})
        if col in test_df.columns:
            test_df[col] = test_df[col].map({'Yes': 1, 'No': 0, 1.0: 1, 0.0: 0})
            
    # Calculate age difference between parents
    train_df['Parent_Age_Diff'] = np.abs(train_df['Mother Age'] - train_df['Father Age'])
    test_df['Parent_Age_Diff'] = np.abs(test_df['Mother Age'] - test_df['Father Age'])

    # Mean of health test results
    test_cols = [f'Test {i}' for i in range(1, 6)]
    train_df['Health_Test_Mean'] = train_df[test_cols].mean(axis=1)
    test_df['Health_Test_Mean'] = test_df[test_cols].mean(axis=1)

    # Total symptom count
    symptom_cols = [f'Symptom {i}' for i in range(1, 6)]
    train_df['Total_Symptom_Count'] = train_df[symptom_cols].sum(axis=1)
    test_df['Total_Symptom_Count'] = test_df[symptom_cols].sum(axis=1)

    # WBC to Blood Cell Ratio
    train_df['WBC_to_BloodCell_Ratio'] = train_df['White Blood cell count'] / train_df['Blood cell count']
    test_df['WBC_to_BloodCell_Ratio'] = test_df['White Blood cell count'] / test_df['Blood cell count']

    # Parental Risk Count
    risk_cols = ['Genes in mother\'s side', 'Genes in father\'s side', 'Maternal gene', 'Paternal gene']
    train_df['Parental_Risk_Count'] = train_df[risk_cols].sum(axis=1)
    test_df['Parental_Risk_Count'] = test_df[risk_cols].sum(axis=1)
    
    # Pregnancy Risk Index (Multiplicative)
    train_df['Pregnancy_Risk_Index_Mult'] = (
        train_df['Mother Age'] * train_df['Father Age'] * (train_df['No. of previous abortion'] + 1)
    )

    test_df['Pregnancy_Risk_Index_Mult'] = (
        test_df['Mother Age'] * test_df['Father Age'] * (test_df['No. of previous abortion'] + 1)
    )

    return train_df, test_df

def scale_features(train_df, test_df, columns_to_scale):
    """Applies StandardScaler to a list of numerical columns."""
    scaler = StandardScaler()
    train_df[columns_to_scale] = scaler.fit_transform(train_df[columns_to_scale])
    test_df[columns_to_scale] = scaler.transform(test_df[columns_to_scale])
    return train_df, test_df, scaler

def drop_and_ohe(df, drop_cols, target_gd=None, target_subclass=None):
    """
    Drops unnecessary columns and applies One-Hot Encoding to the remaining 
    categorical features.
    """
    df_temp = df.copy()
    
    # 1. Drop columns
    # Drop Patient Id and the high NA columns
    cols_to_drop = [c for c in ['Patient Id'] + drop_cols if c in df_temp.columns]
    df_temp.drop(columns=cols_to_drop, errors='ignore', inplace=True)

    # 2. Identify remaining categorical columns
    targets = [t for t in [target_gd, target_subclass] if t is not None]
    
    # Identify non-numeric, non-target columns for OHE
    categorical_cols = df_temp.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c not in targets and c not in ['Blood test result', 'Respiratory Rate', 'Heart Rate']]
    
    # Manually re-add categorical columns that were converted to numeric (e.g., Blood test result, Heart Rate)
    # but still need OHE after conversion, or are still object type.
    ohe_cols = [
        'Status', 'Respiratory Rate', 'Heart Rate', 'Follow-up', 'Gender', 
        'Autopsy shows birth defect (if applicable)', 'Place of birth', 
        'Birth defects', 'Blood test result'
    ]
    categorical_cols = list(set(categorical_cols + [c for c in ohe_cols if c in df_temp.columns]))
    
    # 3. Apply OHE
    df_temp = pd.get_dummies(df_temp, columns=categorical_cols, dummy_na=False, drop_first=True)
    
    # Remove any columns that are now all NaN or inf
    df_temp = df_temp.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='all')

    return df_temp

# --- Main Pipeline Function ---

def process_data_pipeline(train_data, test_data, target_gd, target_subclass, imputation_cols, drop_high_na_cols, scale_cols):
    """
    Executes the full data processing workflow including cleaning, target encoding,
    imputation, feature engineering, OHE, and scaling.

    Args:
        train_data (str): Path to the training data CSV.
        test_data (str): Path to the testing data CSV.
        target_gd (str): Column name for the main genetic disorder target.
        target_subclass (str): Column name for the disorder subclass target.
        imputation_cols (list): Columns to use for MICE imputation.
        drop_high_na_cols (list): Columns with high NaN count to drop.
        scale_cols (list): Numerical columns to scale using StandardScaler.

    Returns:
        tuple: (train_final_df, test_final_df, le_disorder, le_subclass, scaler)
               The fully processed DataFrames and fitted transformers.
    """
    
    # --- Step 0: Load Data ---
    print("Starting data processing pipeline...")
    train_df = load_data(train_data)
    test_df = load_data(test_data)
    
    # --- Step 1: Cleaning and Initial Standardization ---
    train_df = replace_null(train_df)
    test_df = replace_null(test_df)
    
    train_df = rename_columns(train_df)
    test_df = rename_columns(test_df)
    
    # --- Step 2: Target Encoding (Must happen before dropping or OHE) ---
    train_df, test_df, le_disorder, le_subclass = encode_targets(
        train_df, test_df, target_gd, target_subclass
    )
    
    # --- Step 3: Type Conversion and Imputation ---
    train_df = convert_to_float(train_df)
    test_df = convert_to_float(test_df)
    
    train_df, test_df = impute_values(train_df, test_df, imputation_cols)
    
    # --- Step 4: Feature Engineering (Creating new features and standardizing simple bools) ---
    train_df, test_df = feature_engineer(train_df, test_df)

    # --- Step 5: Drop High NA Columns and OHE ---
    train_final = drop_and_ohe(train_df, drop_high_na_cols, target_gd, target_subclass)
    test_final = drop_and_ohe(test_df, drop_high_na_cols)

    # Align columns for train and test after OHE (must run after OHE)
    # We only keep columns that exist in the training set
    train_cols = train_final.drop(columns=[target_gd, target_subclass]).columns
    
    # Reindex test_final to match train_final columns, filling missing with 0
    test_final = test_final.reindex(columns=train_cols, fill_value=0)
    
    # --- Step 6: Feature Scaling ---
    train_final, test_final, scaler = scale_features(train_final, test_final, scale_cols)
    
    print("Data processing pipeline finished successfully.")
    
    return train_final, test_final, le_disorder, le_subclass, scaler