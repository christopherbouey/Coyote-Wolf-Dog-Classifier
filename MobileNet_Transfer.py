import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, TensorBoard
import pathlib
data_dir = pathlib.Path('imgs/train')



#print number of images in each category
image_count = len(list(data_dir.glob('wolf/*.jpg')))
print(f"Wolf: {image_count}")
image_count = len(list(data_dir.glob('coyote/*.jpg')))
print(f"Coyote: {image_count}")
image_count = len(list(data_dir.glob('dog/*.jpg')))
print(f"Dog: {image_count}")

batch_size = 32
img_height = 224
img_width = 224

#set training generator with image adjustment included
train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2) # set validation split

#set testing generator without image adjustment included
test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    seed=123) # set as training data
#setting same random seed should pull same values in each image data generator
validation_generator = test_datagen.flow_from_directory(
    data_dir, # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    seed=123) # set as validation data

#set base model without top
base_model = tf.keras.applications.MobileNet(input_shape = (224,224,3),
                                            include_top=False,
                                            weights='imagenet')

#save basemodel output to variable
x = base_model.output

#set top layers to model (will train these layers)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x=tf.keras.layers.Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=tf.keras.layers.Dense(1024,activation='relu')(x) #dense layer 2
x=tf.keras.layers.Dense(1024,activation='relu')(x) #dense layer 3
x=tf.keras.layers.Dense(512,activation='relu')(x) #dense layer 4
preds=tf.keras.layers.Dense(3,activation='softmax')(x) #final layer with softmax activation

#create model instance
model= tf.keras.models.Model(inputs=base_model.input,outputs=preds)


#freeze base of model
for layer in model.layers[:-6]:
    layer.trainable=False

print(model.summary())


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
mc = ModelCheckpoint(filepath = './Models2/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                    monitor='val_loss',
                    verbose=2,
                    save_best_only=True,
                    save_weights_only=True,
                    mode='min',
                    save_freq='epoch')
#set learning rate reduction
rlrop = ReduceLROnPlateau(monitor='val_loss',
                         factor=0.1,
                         patience=30,
                         verbose=0,
                         mode='auto',
                         min_delta=0.01,
                         cooldown=4,
                         min_lr=10e-8)


model.compile(optimizer=tf.keras.optimizers.Adam(
              learning_rate=0.00001, beta_1=0.9, beta_2=0.999,
              epsilon=1e-07, amsgrad=False,
              name='Adam'),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


step_size_train=train_generator.n//train_generator.batch_size

train_generator.n

train_generator.batch_size

step_size_train

history = model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   epochs=200,
                   validation_data=validation_generator,
                   callbacks=[tb, mc, rlrop])

model.save('Saved_Model')

from matplotlib import pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('acc_val_acc.png')



plt.figure(figsize=(10,5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Categorical Cross Entropy Loss ')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('loss_val_loss.png')

