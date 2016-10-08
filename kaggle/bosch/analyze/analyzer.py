from pandas import DataFrame, read_csv

import pandas as pd
import sys



Location = r'../Data/train_date.csv'
df = pd.read_csv(Location)

print df.head()
