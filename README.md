# ESRGAN

Description: This repository takes from multiple examples of the ESRGAN for image super resolution and implements the model in Tensorflow 2. This is  pretty much the same workflow as the TF_SRGAN repository you can find onn my [GitHub] (https://github.com/dmmagdal/TF_SRGAN).

### How to Use:

 > Install the required modules from requirements.txt with `pip install -r requirements.txt`. The best way to train ESRGAN from scratch is to use the training loop defined in `train.py`. Simply run `python train.py` and it will download the dataset (div2k/bicubic_x4 from tensorflow datasets) and begin training the neural network. You can go inside and alter the training hyperparameters (ie `batch_size`, `epochs`, etc), making this repo very easy to use for training the model from scratch.


### Sources:

 - [GitHub] (https://github.com/SavaStevanovic/ESRGAN)
 - [Medium] (https://medium.com/analytics-vidhya/esrgan-enhanced-super-resolution-generative-adversarial-network-using-keras-a34134b72b77)
 - [Original ESRGAN Paper] (https://arxiv.org/abs/1809.00219)
 - [Paper for Real ESRGAN] (https://arxiv.org/pdf/2107.10833.pdf)
 - [DGGAN Keras Example] (https://keras.io/examples/generative/dcgan_overriding_train_step/)


### Training Photos:

1) A 500 epoch run with the training loop in train.py. This training loop included the following loss functions for the generator: Vgg-MSE (aka content loss) and weighted BCE from discriminator (perceptual loss). Note that there is still some distortion or "glitching" in the images despite being very similar to the high resolution image. 

2) A 500 epoch run with the training loop in train.py. This training loop has the same parameters and configuration as the previous one except the sigmoid activationn funnctionn was added at the end of the discriminator. The previous run was more in line with the original code on Medium/GitHub which did not have the sigmoid activation in the discriminator for some reason. This is generally viewed as  the best training run (between the previous runs here and the best runs from the SRGAN repository).