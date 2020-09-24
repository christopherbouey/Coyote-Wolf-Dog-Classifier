#imports
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, TensorBoard
import pathlib

#set file path to training images
data_dir = pathlib.Path('imgs/train')


def image_counts(categories):
    "Take list of strings of category names and count images in each"
    for i in categories:
        count = len(list(data_dir.glob(i+'/*.jpg')))
        print(f"{i} images: {count}")


def data_set(batch_size, height, width):
    '''Takes batch size, img height, and img width and returns train_generator and validation_generator keras
    objects with data augmentation in the train_generator and a split of 20%'''
    batch_size = batch_size
    img_height = heigh
    img_width = width

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
    return train_generator, validation_generator

def build_model():
    #set base model without top
    base_model = tf.keras.applications.MobileNet(input_shape = (224,224,3),
                                                include_top=False,
                                                weights='imagenet')

    #save basemodel output to variable
    x = base_model.output

    #set top layers to model (will train these layers)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    preds=tf.keras.layers.Dense(3,activation='softmax')(x) #final layer with softmax activation

    #create model instance
    model= tf.keras.models.Model(inputs=base_model.input,outputs=preds)

    #freeze base of model
    for layer in model.layers[:-3]:
        layer.trainable=False

    print(model.summary())

    return model

def set_callbacks(red_patience, early_patience):
    '''takes inputs of patience for learning rate reduction, and early stopping and returns
    tensorboard, modelcheckpoint, reducedlronplateau, and earlystopping callbacks'''
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    mc = ModelCheckpoint(filepath = './Models/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                        monitor='val_loss',
                        verbose=2,
                        save_best_only=True,
                        save_weights_only=True,
                        mode='min',
                        save_freq='epoch')
    #set learning rate reduction
    rlrop = ReduceLROnPlateau(monitor='val_loss',
                            factor=0.1,
                            patience=red_patience,
                            verbose=0,
                            mode='auto',
                            min_delta=0.01,
                            cooldown=4,
                            min_lr=10e-8)
    #early stopping
    es = EarlyStopping(monitor='val_loss',
                    min_delta=0.005,
                    patience=early_patience,
                    verbose=1,
                    mode='min',
                    restore_best_weights=True)
    return tb, mc, rlrop, es

def model_compile(model, LR=0.0001, loss_='categorical_crossentropy', metrics_=['accuracy']):
    '''Takes float of learning rate, string of loss method, and list of metrics and compiles model with adam optimizer'''
    model.compile(optimizer=tf.keras.optimizers.Adam(
                learning_rate=LR, beta_1=0.9, beta_2=0.999,
                epsilon=1e-07, amsgrad=False,
                name='Adam'),
                loss=loss_,
                metrics=metrics_)

def fit_save_model(epochs_=200, callbacks_= None, train_generator, validation_generator, save_name):
    '''Takes the model object, number of epochs, callback list (from callback function), training data generator, and validation data generator
    and fits the model and returns the history of the training'''
    step_size_train=train_generator.n//train_generator.batch_size

    history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=step_size_train,
                    epochs=200,
                    validation_data=validation_generator,
                    callbacks=[tb, mc, rlrop, es])

    model.save(save_name)
    return history

def graph_history(metrics, loss_type, hist):
    '''Takes in list of metrics of interest that were calculated, string object naming loss type, and
    history object (returned from fit_save_model()) and returns graphs of metric history and loss history'''
    for i in metrics:
        from matplotlib import pyplot as plt
        plt.figure(figsize=(10,5))
        plt.plot(hist.history[i])
        plt.plot(hist.history['val_'+i])
        plt.title('Model '+i)
        plt.ylabel(i)
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(i+'_graph.png')



        plt.figure(figsize=(10,5))
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title(loss_type+' Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('loss_graph.png')

def main():
    image_counts(['coyote','dog','wolf'])
    train_generator, validation_generator = data_set(32, 224, 224)
    model = build_model()
    tb, mc, rlrop, es = set_callbacks(10,30)
    model_compile(model)
    history = fit_save_model(callbacks_= [mc, rlrop, es], train_generator, validation_generator, 'Saved_Model')
    graph_history(['accuracy'], 'Categorical Cross Entropy'), history)

if __name__ == "__main__":
    main()

