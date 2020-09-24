# Coyote-Wolf-Dog Classifier

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/christopherbouey/Coyote-Wolf-Dog-Classifier">
    <img src="Images/wolf-vs-coyote.webp" alt="Logo" width="460" height="260">
  </a>

  <h3 align="center">Comparison Image/</h3>
</p>



<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Description](#about-the-project)
* [Data Exploration](#data-exploration)
* [Pipeline](#pipeline)
* [Transfer Learning](#transfer-learning)
* [Results](#results)
* [Next Steps](#next-steps)




<!-- ABOUT THE PROJECT -->
## Description

The goal of this project was to develop a fine-grained image classifier that could differentiate between coyotes, domesticated dogs, and wolves successfully.

I wanted an opportunity to explore Transfer Learning and Image Classification. In my search for an available dataset and a project purpose, I remembered a recent [article](https://www.sfgate.com/living-in-sf/article/New-footage-Coyotes-take-to-SF-rooftops-frolick-15257171.php#item-95844-tbla-2) about a resurgence of coyote activity in San Francisco thanks to the recent decrease in human activity. I have always had trouble distinguishing between coyotes and other canines, so I decided to build an image classifier to classify pictures of coyotes, wolves, and domesticated dogs (not that there will be wolves coming into SF any time soon, but I wanted to gain experience in multi-class models).




## Data Overview
My data was taken from Imagenet, an open source image library. It is structured such that by doing an API call with a certain ID, you can access a list of url links to photos of that ID item.

From Imagenet, I loaded in every available photo from the following species IDs:  n02114855|Coyote, n02084071|Dog, n02114367|Wolf

There were many links available for each category on Imagenet's site
* Coyote: 1664
* Dog: 1603
* Wolf: 1820

---

## Data Exploration

#### Cleaned Data
To be expected, some of the links were no longer active, and some of the photos were smaller than my resize shape (see below), and so not each link produced a usable image.

After resizing photos to (224,224,3), I was able to download the following number of useable photos
* Coyote: 911
* Dog: 771
* Wolf: 862

<p align="center">
  <img align="center" src="Images/Class_dist.png" width="400" height="400" />
</p>

Here is an example of each after resizing:

<center>
  
Coyote          |  Dog                      |        Wolf
:-------------------------:|:-------------------------:|:-------------------------:
![](Images/coyote_1.jpg)  |  ![](/Images/dogs_10.jpg)  |   ![](/Images/wolf_105.jpg)

 </center>

---

## Pipeline
### TensorFlow Directory Structure

I utilized tensorflow's flow_from_directory method to load in my photos. Because of this, I had to be sure to load my photos in with the following format with the photos saved as jpg files:

    |-Images 

         |-Coyote

         |-Dog
         
         |-Wolf

### Data Augmentation

To increase the effectivenss of the relatively small train set, I used TF's ImageDataGenerator it set to adjust the shear, rotation, zoom, and a horizontal flip of the photos randomly as they are called on when training the model. 

This method serves to avoid overfitting by artificially gernerating "new" images so that the model can capture angles/positions/sizes of the objects that do not actually exist in the training set. 

### Callbacks

To increase the efficiency and effectiveness of the model, I utilized the following callbacks:
* ModelCheckpoint
* ReducedLROnPlateau
* EarlyStopping

## Transfer Learning

### Base

Because of the limited timeline to perform this project, I used the lightweight MobileNetV2 model with the Imagenet weights loaded as the base model. I removed the top from this model to append my own classifier on top. The model was pre-trained on the full Imagenet dataset and the weights were imported and applied to the base model. These weights were then frozen to maintain the feature extraction capabilities gained in the original training.

### Classifying Layers

Structure:
1. Global Average Pooling
2. 3 Fully Connected Dense Layers
3. One half reduction Dense Layer
4. Classifying Dense layer

Average pooling helps to prevent overfitting by averaging the vectorized images. In concept, this keeps the model from focusing on the location of features in the images

---

## Results

### Prediction
The following image is based on a 16 size batch of validation data:
![grid_predictions](/Images/prediction_photo_array.jpg)

14 out of 16 were correctly labeled, with one photo in the second row being mislabeled 'wolf' instead of 'dog', and an image in the last row being mislabeled 'wolf' instead of 'coyote'

### Metrics
**Accuracy:**

Initial: ~0.78

![](/Images/early_acc_graph.png)

After Fine Tuning: ~0.81

![](/Images/fine_tune_acc.png)

**Categorical Cross Entropy Loss**

Initial: ~0.64

![](/Images/early_loss_graph.png)

After Fine Tuning: ~0.53

![](/Images/fine_tune_loss.png)


## Next Steps
-Harvest data from instagram to test model on

-Test out other classification architectures

-Test out other pre-trained models

-Save augmented images into drive to perform preliminary feature extraction : will significantly reduce training time in future

-Try more fine-tuning of the model

[Back To The Top]
