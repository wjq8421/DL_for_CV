import matplotlib
from callbacks.trainingmonitor import TrainingMonitor
from nn.conv.minivggnet import MiniVGGNet
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import SGD
from keras.datasets import cifar10
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True, help='path to the output directory')
args = vars(ap.parse_args())

# If training going poorly, Simple open up task manager and
# kill the process ID associated with my script
print("[INFO] process ID: {}".format(os.getpid()))

print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype('float') / 255.0
testX = testX.astype('float') / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)
labelNames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print("[INFO] compiling network...")
opt = SGD(lr=0.01, decay=0.01/40, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

figPath = "{}_{}.png".format(args['output'], os.getpid())
jsonPath = "{}_{}.json".format(args['output'], os.getpid())
callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath)]

print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=64, epochs=100, verbose=1, callbacks=callbacks)
