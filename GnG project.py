import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
                    'Patient ID',
                    'Father\'s Name',
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



















