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
- We have trained 25 different models on our training set, then evaluate them on the test set to pick out the ones that have the lowest loss: 

* Model 1: VGG19
* Model 2: Inception ResNetV2
* Model 3: DenseNet201

- Hyper-parameters that we use during our training session:


## 3. Ensemble models
- We use the GridSearch-based algorithms to find the smallest loss possible when ensembling three models.
- Here, we let the coefficient of VGG19’s predictions, Inception Resnet V2’s predictions, and Densenet 201 predictions run from 0 to 1, the step is 0.1, and find the best fit with minimum loss.

* Best weight for VGG19: 0.2
* Best weight for Inception Resnet V2: 0.4
* Best weight for  DenseNet201: 0.4

## 4. Deploy models
* We have deployed the model with Flask and hosted it on Heroku.
* Link: https://aiijcdraft.herokuapp.com/


