from nn.conv.lenet import LeNet
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras import datasets
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

print("[INFO] loading MNIST (full) dataset...")
(trainX, trainY), (testX, testY) = datasets.mnist.load_data()

if K.image_data_format() == 'channels_first':
    trainX = trainX.reshape(trainX.shape[0], 1, 28, 28)
    testX = testX.reshape(testX.shape[0], 1, 28, 28)
else:
    trainX = trainX.reshape(trainX.shape[0], 28, 28, 1)
    testX = testX.reshape(testX.shape[0], 28, 28, 1)

print(trainX.shape)
print(trainY.shape)

trainX = trainX.astype('float') / 255.0
testX = testX.astype('float') / 255.0
(trainX, valX, trainY, valY) = train_test_split(trainX, trainY, test_size=0.25)

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
valY = lb.fit_transform(valY)
testY = lb.fit_transform(testY)

print("[INFO] compiling network...")
sgd = SGD(lr=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(valX, valY), batch_size=128, epochs=100, verbose=1)
model.save('lenet_mnist.h5')

print('[INFO] evaluating network...')
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=[str(x) for x in lb.classes_]))