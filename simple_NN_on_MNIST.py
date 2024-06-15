# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 21:26:07 2024

@author: gbulb
"""

from tensorflow import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255
from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers 
from keras import layers
###########
class simple_NN(Sequential):
    def __init__(self,params):
        super().__init__()
        self.add(layers.Dense(512, activation='relu', input_shape=params["target_shape"]))
        self.add(layers.Dense(params["num_classes"], activation=params["activation"]))
        self.summary()
        self.compile(optimizer=params["optimizer"],
        loss=params["loss"],
        metrics=params["metric"])


params={"target_shape":(28 * 28,),
        "activation":'softmax',
        "optimizer":'rmsprop',
        "loss":'categorical_crossentropy',
        "metric":'accuracy',
        "num_classes":10
}
network=simple_NN(params)
callbacks_list = [
keras.callbacks.EarlyStopping(
monitor=params["metric"],
patience=1,
),
keras.callbacks.ModelCheckpoint(
filepath='my_model.h5',
monitor='val_loss',
save_best_only=True,
)
]
x_val = train_images[-10000:]
y_val = train_labels[-10000:]
x_train = train_images[:-10000]
y_train = train_labels[:-10000]
history=network.fit(x_train,y_train,callbacks=callbacks_list,validation_data=(x_val, y_val),epochs=10)
test_loss, test_acc = network.evaluate(test_images, test_labels)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix
predictions = network.predict(test_images)
cm = confusion_matrix(test_labels.argmax(axis=1), predictions.argmax(axis=1))
cm
cm_plot_labels = range(1,11,1)
import matplotlib.pyplot as plt
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
# plot accuracy during training
plt.subplot(212)
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.legend()
plt.show()

from tensorflow.keras.utils import plot_model
plot_model(network, "simple_NN_on_MNIST.png", show_shapes=True)