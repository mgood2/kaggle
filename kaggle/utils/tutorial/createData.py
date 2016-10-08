from pandas import DataFrame, read_csv

import matplotlib.pyplot as pyplot
import pandas as pd
import sys
import matplotlib


names = ['Bob','Jessica','Mary','John','Mel']
births = [968, 155, 77, 578, 973]

BabyDataSet = list(zip(names,births))

print BabyDataSet

#DataFrame

df = pd.DataFrame(data = BabyDataSet, columns=['Names', 'Births'])

print df

df.to_csv('births1880.csv', index=False, header=False)
