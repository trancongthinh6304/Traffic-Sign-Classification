# trafficsignclassification

## TABLE OF CONTENT
1. Prepocess image
2. Train models
3. Ensemble models
4. Deploy models

## 1. Preprocess image
### a. Autoencoder model (Denoise image)
* In order to improve image resolution, we trained an auto encoder model.
* Training data: 6000 high resolution image and 6000 low resolution image
* Input size: 80x80
### b. Image augmentation
* To increase the training data, we augment each image in five ways

## 2. Train models
* Model 1: VGG19
* Model 2: Inception ResNetV2
* Model 3: DenseNet201
* Abundant ones

## 3. Ensemble models
* Best weight for VGG19: 0.2
* Best weight for Inception Resnet V2: 0.4
* Best weight for  DenseNet201: 0.4

## 4. Deploy models
* We have deployed the model with Flask and hosted it on Heroku.
* Link: https://aiijcdraft.herokuapp.com/


