from pandas import DataFrame, read_csv

import pandas as pd
import sys



Location = r'C:/Users/elee/Documents/workspace/kaggle/kaggle/bosch/Data/train_date.csv'
df = pd.read_csv(Location)

print df.head()
