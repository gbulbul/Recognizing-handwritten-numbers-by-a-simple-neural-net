# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 23:44:29 2024

@author: gbulb
"""

from tensorflow import keras
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers 
from keras import layers
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.datasets import mnist
####Loading the dataset#####
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#####Preprocessing##########
###PART I(arranging images)#########
img_rows, img_cols=28, 28
num_channel=1
train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols,num_channel).astype('float32') / 255
test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols,num_channel).astype('float32') / 255
###PART II(arranging labels)#########
from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
###########CNN Modeling#################
class CNN(Sequential):
    def __init__(self,params):
        super().__init__()
        ###########
        self.add(Conv2D(32, (3, 3), activation=params["activation_re"], kernel_initializer=params["initializer"], input_shape=params["target_shape"]))
        self.add(MaxPooling2D((2, 2)))
        self.add(Flatten())
        self.add(Dense(100, activation=params["activation_re"], kernel_initializer=params["initializer"]))
        self.add(Dense(params["num_classes"], activation=params["activation_so"]))
        self.summary()
        self.compile(optimizer=params["optimizer"], loss=params["loss"], metrics=params["metric"])
params={"target_shape":(img_rows, img_cols,num_channel),
        "activation_so":'softmax',
        "activation_re":'relu',
        "initializer":'he_uniform',
        "optimizer":SGD(learning_rate=0.01, momentum=0.9),
        "loss":'categorical_crossentropy',
        "metric":'accuracy',
        "num_classes":10}
model=CNN(params)
history = model.fit(xtrain, train_labels, validation_split=0.1,epochs=10)
########Model evaluation & diagnostics##########
test_loss, test_acc = model.evaluate(xtest, test_labels)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix
predictions = model.predict(xtest)
cm = confusion_matrix(test_labels.argmax(axis=1), predictions.argmax(axis=1))
cm

cm_plot_labels = range(1,11,1)

fig, ax = plot_confusion_matrix(conf_mat=cm, class_names=cm_plot_labels, colorbar=True)
plt.show()

fig = plt.figure(figsize=(12, 12))
plt.subplot(211)
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()

plt.subplot(212)
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.legend()
plt.show()
######Model visualization#######
from tensorflow.keras.utils import plot_model
plot_model(model, 'CNN.png',show_shapes=True)
