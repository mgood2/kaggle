from pandas import DataFrame, read_csv

import matplotlib.pyplot as pyplot
import pandas as pd
import sys
import matplotlib

Location = r'C:/Users/elee/Documents/workspace/kaggle/kaggle/utils/tutorial/births1880.csv'
df = pd.read_csv(Location, names=['Names', 'Births'])

Sorted = df.sort_values(['Births'], ascending=False)
print Sorted.head(1)

print df['Births'].max()
