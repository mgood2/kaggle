import numpy as np
import pandas as pd

print("Reading data...")
dataset = pd.read_csv("../data/train.csv.old")
target = dataset[[0]].values.ravel()
train = dataset.iloc[:,1:].values
test = pd.read_csv("../data/test.csv.old").values

#rf = RandomForestClassifier(n_estimators=100)
#rf.fit(train,target)
#pred = rf.predict(test)
#np.savetxt('submission_rand_forest.cvs', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImagedId,Label', comments= '', fmt='%d')
#accuracy ~ 0.96

target = target.astype(np.uint8)
train = np.array(train).reshape((-1,1,28,28)).astype(np.unit8)
test = np.array(test).reshape((-1,1,28,28)).astype(np.uint8)



