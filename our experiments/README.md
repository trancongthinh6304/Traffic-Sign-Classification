1. Our GANs:
Primary goal: to detect unknown images
Method: 
* Outline: We train and let our discriminator and our generator models against each other in the training loop. Since the goal of the generator is to create fake images from noise to trick the discriminator, and the discriminator’s goal is to detect fake images from real images, we theorize that by putting a powerful generator against our discriminator, our discriminator may be able to distinguish fake images, hence detecting unknown images. 
* For the discriminator model: we feed it with a batch of 64 fake images (created by the generator model) and a batch of 64 real images (images from the training set that does not belong to the “unknown” class). The result (how well the discriminator classifies the fake and real images) will be used as feedback for the generator model.
* For the generator model: the generator will create a batch of 64 fake images (from noise) to feed to the discriminator model.
Result: our generator is not powerful enough to create convincing images to trick the discriminator. Hence, our discriminator ultimately could not detect fake images well enough

2. Our one-class Autoencoder model:
Primary goal: to detect unknown images
Method: 
* Outline: we create and train an autoencoder model on the training set (without unknown images) with the ultimate goal is to utilize its loss to detect unknown images.
* Theory: The mechanism of our autoencoder model is to compare the original images to the output images (of the decoder part) to compute the mean squared error/binary cross-entropy loss. If we, later on, use this model to predict unknown images, then compute the loss → the unknown images will have a bigger loss compared to traffic sign images → we can use this as a sign to detect unknown images
Result: the loss of the unknown images is almost the same as the loss of the traffic sign images. We could not use the loss to detect unknown images

