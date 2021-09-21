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
   - Model 1: VGG19
   - Model 2: Inception ResNetV2
   - Model 3: DenseNet201

- Hyper-parameters that we use during our training session:


## 3. Ensemble models
- We use the GridSearch-based algorithms to find the smallest loss possible when ensembling three models.
- Here, we let the coefficient of VGG19’s predictions, Inception Resnet V2’s predictions, and Densenet 201 predictions run from 0 to 1, the step is 0.1, and find the best fit with minimum loss.

  - Best weight for VGG19: 0.2
  - Best weight for Inception Resnet V2: 0.4
  - Best weight for  DenseNet201: 0.4

## 4. Deploy models
### a. Web application
* We have deployed the model with Flask and hosted it on Heroku.
* Link: https://aiijcdraft.herokuapp.com/

### b. Docker
1. Open your browser to https://labs.play-with-docker.com/
2. Click Login and then select docker from the drop-down list.
3. Connect with your Docker Hub account.
4. Once you’re logged in, click on the ADD NEW INSTANCE option on the left side bar. If you don’t see it, make your browser a little wider. After a few seconds, a terminal window opens in your browser.![](https://i.imgur.com/3JqCAEI.png)

5. In the terminal, start your freshly pushed app.
 docker run -dp 5000:5000 vvai1710/babyshark-aiijc2021:app
6. Click on the 5000 badge when it comes up and you should see the app with your modifications! Hooray! If the 5000 badge doesn’t show up, you can click on the “Open Port” button and type in 5000.![](https://i.imgur.com/2RR1lbz.png)


