import pandas as pd
import sys


d = [0,1,2,3,4,5,6,7,8,9]

df = pd.DataFrame(d)


# Lets change the name of the column
df.columns = ['Rev']

# Lets add a column
df['NewCol'] = 5
# Lets modify our new column
df['NewCol'] = df['NewCol'] + 1


# We can delete columns
del df['NewCol']

# Lets add a couple of columns
df['test'] = 3
df['col'] = df['Rev']


# If we wanted, we could change the name of the index
i = ['a','b','c','d','e','f','g','h','i','j']
df.index = i

print df.loc['a']
