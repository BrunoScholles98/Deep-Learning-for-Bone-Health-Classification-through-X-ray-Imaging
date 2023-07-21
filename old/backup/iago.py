# -*- coding: utf-8 -*-
#resnet50.ipynb

from datetime import datetime
import pathlib
from pathlib import Path
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow import keras
from tensorflow.keras import layers

#!gdown 1qUoghA1lCAketG8cMFCG_tQ6jX7Dbq_z

#!unzip /content/data_folder.zip -d /content/data_folder/

#from datetime import datetime

start_time = datetime.now()

#from pathlib import Path

print("DADOS DE NOME DO ARQUIVO")

#get own file name
file_code = "string"
file_code = Path(__file__).stem
print(file_code)
print("\n")

#from datetime import datetime

print("DADOS DE DATA")

# datetime object containing current date and time
now = datetime.now()
 
print("now =", now)

# dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)

dt_code = now.strftime("%Y%m%d_%H%M%S")
print("date code =", dt_code)
print(type(dt_code))
print("\n")

print("DATASET E MODELOS DIR")
dataset_dir = 'data_folder'
print(dataset_dir)
filename = file_code + '_' + dt_code
print(filename)
models_dir = 'models_1024/' + filename
print(models_dir)

os.makedirs(models_dir, exist_ok=True) 
os.makedirs(dataset_dir, exist_ok=True) 

print("\n")

#import pathlib

new_base_dir = pathlib.Path(dataset_dir)

#import os

train_dir = os.path.join(new_base_dir, 'train')
validation_dir = os.path.join(new_base_dir, 'val')
test_dir = os.path.join(new_base_dir, 'test')

path = dataset_dir + '/test/Healthy'
files = os.listdir(path)

#from PIL import Image

print("TAMANHO DAS IMAGENS")

image = Image.open(path + '/' + files[0])
print(image.size)
w, h = image.size
print('width: ', w)
print('height:', h)

resized = (int(w/4), int(h/4))
resized = list(resized)
print(resized)
print("\n")

#from tensorflow.keras.utils import image_dataset_from_directory

train_dataset = image_dataset_from_directory(
    new_base_dir / "train",
    image_size=resized,
    batch_size=4)
validation_dataset = image_dataset_from_directory(
    new_base_dir / "val",
    image_size=resized,
    batch_size=4)
test_dataset = image_dataset_from_directory(
    new_base_dir / "test",
    image_size=resized,
    batch_size=4)

print("IMPORTAR E CONGELAR REDE")

conv_base  = keras.applications.resnet.ResNet50(
    weights = "imagenet",
    include_top = False)
conv_base.trainable = False

conv_base.trainable = True
print("This is the number of trainable weights "
      "before freezing the conv base:", len(conv_base.trainable_weights))

conv_base.trainable = False
print("This is the number of trainable weights "
      "after freezing the conv base:", len(conv_base.trainable_weights))
print("\n")

for i, layer in enumerate(conv_base.layers):
   print(i, layer.name, layer.trainable)
print("\n")

conv_base.summary()

#from tensorflow import keras
#from tensorflow.keras import layers

print("GERAR MODELO")

inputs = keras.Input(shape=(resized[0], resized[1], 3))
x = keras.applications.resnet.preprocess_input(inputs)
x = conv_base(x)
x = layers.Flatten()(x)
x = layers.Dense(128)(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256)(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(512)(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(1024)(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()

from keras.utils.vis_utils import plot_model

dot_img_file = models_dir + '/' + filename + '_model.png'
#keras.utils.plot_model(model, to_file = dot_img_file, show_shapes = True)

print("\n")

dot_img_file = models_dir + '/' + filename + '_img.png'
dot_img_file

print("COMPILAR E TREINAR")

model.compile(loss="binary_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath = models_dir + '/' + filename + '_frozen.keras',
        save_best_only = True,
        monitor = "val_loss")
]
history = model.fit(
    train_dataset,
    epochs = 80,
    validation_data = validation_dataset,
    callbacks = callbacks,
    verbose = 2)

print(history.params)
print("\n")

#import numpy as np

np.save(models_dir + '/' + filename + '_frozen_history.npy',history.history)

#import matplotlib.pyplot as plt

dpi = 96
height = 800
width = 500

accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)

plt.figure(figsize=(height/dpi, width/dpi), dpi = dpi)
plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and Validation accuracy")
plt.legend()
plt.savefig(models_dir + '/' + filename + '_frozen_accuracy.png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(height/dpi, width/dpi), dpi = dpi)
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and Validation loss")
plt.legend()
plt.savefig(models_dir + '/' + filename + '_frozen_loss.png', bbox_inches='tight')
plt.show()

print("\n")

print("PREDICAO")

test_model = keras.models.load_model(models_dir + '/' + filename + '_frozen.keras')
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Frozen test accuracy: {test_acc:.3f}")
print("\n")

f = open(models_dir + '/' + filename + '_eval.txt', "a")
f.write(f"Frozen test accuracy: {test_acc:.3f}\n")
f.close()

print("DESCONGELAR REDE")

conv_base.trainable = True
#for layer in conv_base.layers[:165]:
#    layer.trainable = False

print("This is the number of trainable weights "
      "after unfreezing the conv base:", len(conv_base.trainable_weights))
print("\n")

for i, layer in enumerate(conv_base.layers):
   print(i, layer.name, layer.trainable)
print("\n")

print("COMPILAR E TREINAR")

model.compile(loss="binary_crossentropy",
              optimizer=keras.optimizers.RMSprop(learning_rate=1e-5),
              metrics=["accuracy"])

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath = models_dir + '/' + filename + '_unfrozen.keras',
        save_best_only = True,
        monitor = "val_loss")
]
history = model.fit(
    train_dataset,
    epochs = 80,
    validation_data = validation_dataset,
    callbacks = callbacks,
    verbose = 2)

print(history.params)
print("\n")

np.save(models_dir + '/' + filename + '_unfrozen_history.npy',history.history)

#import matplotlib.pyplot as plt

accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)

plt.figure(figsize=(height/dpi, width/dpi), dpi = dpi)
plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Unfrozen Training and Validation accuracy")
plt.legend()
plt.savefig(models_dir + '/' + filename + '_unfrozen_accuracy.png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(height/dpi, width/dpi), dpi = dpi)
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Unfrozen Training and Validation loss")
plt.legend()
plt.savefig(models_dir + '/' + filename + '_unfrozen_loss.png', bbox_inches='tight')
plt.show()

print("\n")

print("PREDICAO")

test_model = keras.models.load_model(models_dir + '/' + filename + '_unfrozen.keras')
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Unfrozen test accuracy: {test_acc:.3f}")
print("\n")

f = open(models_dir + '/' + filename + '_eval.txt', "a")
f.write(f"Unfrozen test accuracy: {test_acc:.3f}\n")
f.close()

now = datetime.now()
print("now =", now)
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)
print("\n")

print('Duration: {}'.format(now - start_time))
print("FIM")
print("\n")