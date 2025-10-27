import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split

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

train_df = train_df.replace({'N/A': np.nan, 'null': np.nan, '-99': np.nan, '-': np.nan, 'Not applicable': np.nan, 'nan': np.nan}) 
test_df = test_df.replace({'N/A': np.nan, 'null': np.nan, '-99': np.nan, '-': np.nan, 'Not applicable': np.nan, 'nan': np.nan})

#print(train_df.dtypes)
#print(test_df.dtypes)

train_df = train_df.rename(columns={'Mother\'s age': 'Mother Age', 'Father\'s age': 'Father Age', 'Heart Rate (rates/min': 'Heart Rate', 'Respiratory Rate (breaths/min)': 'Respiratory Rate'})
test_df = test_df.rename(columns={'Mother\'s age': 'Mother Age', 'Father\'s age': 'Father Age', 'Heart Rate (rates/min': 'Heart Rate', 'Respiratory Rate (breaths/min)': 'Respiratory Rate'})

train_df = train_df.astype({'Patient Age': 'float64', 'Mother Age': 'float64', 'Father Age': 'float64', 'Test 1': 'float64', 'Test 2': 'float64', 'Test 3': 'float64', 'Test 4': 'float64', 'Test 5': 'float64', 'Symptom 1': 'float64', 'Symptom 2': 'float64', 'Symptom 3': 'float64', 'Symptom 4': 'float64', 'Symptom 5': 'float64', 'No. of previous abortion': 'float64'})
test_df = test_df.astype({'Patient Age': 'float64', 'Mother Age': 'float64', 'Father Age': 'float64', 'Test 1': 'float64', 'Test 2': 'float64', 'Test 3': 'float64', 'Test 4': 'float64', 'Test 5': 'float64', 'Symptom 1': 'float64', 'Symptom 2': 'float64', 'Symptom 3': 'float64', 'Symptom 4': 'float64', 'Symptom 5': 'float64', 'No. of previous abortion': 'float64'})


train_df = train_df.dropna(subset=['Genetic Disorder', 'Disorder Subclass'])

imputer = IterativeImputer(max_iter=10, random_state=0)

train_df_imputed = imputer.fit_transform(train_df)







