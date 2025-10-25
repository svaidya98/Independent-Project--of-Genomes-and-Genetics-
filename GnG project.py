import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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






