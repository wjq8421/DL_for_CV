from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor
from datasets.simpledatasetloader import SimpleDatasetLoader
from nn.conv.shallownet import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to input dataset')
args = vars(ap.parse_args())

print('[INFO] loading images...')
imagePaths = list(paths.list_images(args['dataset']))

sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, label) = sdl.load(imagePaths, verbose=100)
data = data.astype('float') / 255.0
label = label.reshape(-1, 1)

(trainX, valX, trainY, valY) = train_test_split(data, label, test_size=0.25, random_state=42)
trainY = OneHotEncoder().fit_transform(trainY)
valY = OneHotEncoder().fit_transform(valY)
print(valY.shape)

print('[INFO] compiling model...')
opt = SGD(lr=0.005)
model = ShallowNet.build(width=32, height=32, depth=3, classes=2)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

print('[INFO] training network...')
H = model.fit(trainX, trainY, validation_data=(valX, valY), batch_size=32, epochs=100, verbose=1)

print('[INFO] evaluating network...')
predictions = model.predict(valX, batch_size=32)
print(classification_report(valY.argmax(axis=1), predictions.argmax(axis=1), target_names=['cat', 'dog']))

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 100), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, 100), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, 100), H.history['acc'], label='train_acc')
plt.plot(np.arange(0, 100), H.history['val_acc'], label='val_acc')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()