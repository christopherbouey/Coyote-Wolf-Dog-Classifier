import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pathlib
data_dir = pathlib.Path('imgs/train')

batch_size = 16
img_height = 224
img_width = 224

#set generator for predictions
test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2) # set validation split
validation_generator = test_datagen.flow_from_directory(
    data_dir, # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    seed=123) # set as validation data



#instantiate model
base_model = tf.keras.applications.MobileNet(input_shape = (224,224,3),
                                            include_top=False,
                                            weights="imagenet")
x = base_model.output

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Flatten()(x)
preds=tf.keras.layers.Dense(3,activation='softmax')(x) #final layer with softmax activation

model= tf.keras.models.Model(inputs=base_model.input,outputs=preds)

base_model.input

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
                name='Adam'),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#load weights
weights_ = input("Enter Filepath to Model Weights: ")
model.load_weights(f'{weights_}')

#save instance of generator
first_gen = validation_generator[0]
pred = model.predict(first_gen)

#set array to boolean class into
animals = np.array(list(validation_generator.class_indices.keys()))

#find prediction of each image
preds_val = np.argmax(pred, axis=1)
pred_animal = animals[preds_val]

#graph the predicions with the images
fig, ax = plt.subplots(nrows=4,ncols=4, figsize=(20,20))
for i, axes in enumerate(ax.flatten()):
    axes.imshow(first_gen[0][i])
    axes.set_title(pred_animal[i],fontdict={'fontsize': 30, 'fontweight': 'medium'})
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)

plt.tight_layout()
plt.savefig('fine_prediction_photo_array.jpg')
