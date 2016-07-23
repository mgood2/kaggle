from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import numpy as np



print('-'*30)
print('Loading train data...')
print('-'*30)
# load pima indians dataset
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]



print('-'*30)
print('Creating and compiling model...')
print('-'*30)

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# save trained model to model.hdf5
model_checkpoint = ModelCheckpoint('model.hdf5', monitor='loss', save_best_only=True)

print('-'*30)
print('Fitting model...')
print('-'*30)
# Fit the model
model.fit(X, Y, nb_epoch=100, batch_size=10, callbacks=[model_checkpoint])

# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



print('-'*30)
print('Loading test data...')
print('-'*30)
# load pima indians dataset
dataset = np.loadtxt("pima-indians-diabetes-test-set.csv", delimiter=",")
# split into input (X) and output (Y) variables
X_test = dataset[:,0:8]

print('-'*30)
print('Loading saved weights...')
print('-'*30)
model.load_weights('model.hdf5')

print('-'*30)
print('Predicting test data...')
print('-'*30)
predicted_labels = model.predict(X_test)

np.savetxt('submission.csv',
           np.c_[range(1,len(X_test)+1),predicted_labels],
           delimiter=',',
           header = '',
           comments = '',
           fmt='%d %10.3f')
