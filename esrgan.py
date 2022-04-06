# esrgan.py
# Implement and train the Enhanced Super Resolution GAN in Tensorflow 2
# for image super resolution.
# Source (GitHub): https://github.com/SavaStevanovic/ESRGAN
# Source (Medium): https://medium.com/analytics-vidhya/esrgan-enhanced-
# super-resolution-generative-adversarial-network-using-keras-
# a34134b72b77
# Source (Paper for ESRGAN): https://arxiv.org/abs/1809.00219
# Source (Paper for Real ESRGAN): https://arxiv.org/pdf/2107.10833.pdf
# Tensorflow 2.7.0
# Windows/MacOS/Linux
# Python 3.7


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG19
import tensorflow_datasets as tfds


class DenseBlock(layers.Layer):
	#def __init__(self, filters=64, kernel_size=3, strides=1, **kwargs):
	def __init__(self, filters=64, kernel_size=(3, 3), **kwargs):
		super().__init__()

		'''
		self.conv1 = layers.Conv2D(
			filters, kernel_size=3, strides=1, padding="same"
		)
		self.leakyReLU1 = layers.LeakyReLU(alpha=0.2)
		self.concat1 = layers.Concatenate()

		self.conv2 = layers.Conv2D(
			filters, kernel_size=3, strides=1, padding="same"
		)
		self.leakyReLU2 = layers.LeakyReLU(alpha=0.2)
		self.concat2 = layers.Concatenate()

		self.conv3 = layers.Conv2D(
			filters, kernel_size=3, strides=1, padding="same"
		)
		self.leakyReLU3 = layers.LeakyReLU(alpha=0.2)
		self.concat3 = layers.Concatenate()

		self.conv4 = layers.Conv2D(
			filters, kernel_size=3, strides=1, padding="same"
		)
		self.leakyReLU4 = layers.LeakyReLU(alpha=0.2)
		self.concat4 = layers.Concatenate()

		self.conv5 = layers.Conv2D(
			filters, kernel_size=3, strides=1, padding="same"
		)
		'''
		self.conv1 = layers.Conv2D(
			filters, kernel_size=(3, 3), padding="same"
		)
		self.leakyReLU1 = layers.LeakyReLU(alpha=0.2)
		self.concat1 = layers.Concatenate()

		self.conv2 = layers.Conv2D(
			filters, kernel_size=(3, 3), padding="same"
		)
		self.leakyReLU2 = layers.LeakyReLU(alpha=0.2)
		self.concat2 = layers.Concatenate()

		self.conv3 = layers.Conv2D(
			filters, kernel_size=(3, 3), padding="same"
		)
		self.leakyReLU3 = layers.LeakyReLU(alpha=0.2)
		self.concat3 = layers.Concatenate()

		self.conv4 = layers.Conv2D(
			filters, kernel_size=(3, 3), padding="same"
		)
		self.leakyReLU4 = layers.LeakyReLU(alpha=0.2)
		self.concat4 = layers.Concatenate()

		self.conv5 = layers.Conv2D(
			filters, kernel_size=(3, 3), padding="same"
		)
		self.lambdaLayer = layers.Lambda(lambda x:x * 0.2)
		self.add = layers.Add()


	def call(self, inputs):
		x1 = self.conv1(inputs)
		x1 = self.leakyReLU1(x1)
		x1 = self.concat1([inputs, x1])

		x2 = self.conv2(x1)
		x2 = self.leakyReLU2(x2)
		x2 = self.concat1([inputs, x1, x2])

		x3 = self.conv3(x2)
		x3 = self.leakyReLU3(x3)
		x3 = self.concat1([inputs, x1, x2, x3])

		x4 = self.conv4(x3)
		x4 = self.leakyReLU4(x4)
		x4 = self.concat1([inputs, x1, x2, x3, x4])

		x5 = self.conv5(x4)
		x5 = self.lambdaLayer(x5)
		outs = self.add([inputs, x5])
		return outs


class RRDBlock(layers.Layer):
	def __init__(self, **kwargs):
		super().__init__()

		self.block1 = DenseBlock()
		self.block2 = DenseBlock()
		self.block3 = DenseBlock()
		self.lambdaLayer = layers.Lambda(lambda x:x * 0.2)
		self.add = layers.Add()


	def call(self, inputs):
		x = self.block1(inputs)
		x = self.block2(x)
		x = self.block3(x)
		x = self.lambdaLayer(x)
		outs = self.add([inputs, x])
		return outs


'''
def generator(inputs):
	x = layers.Conv2D(
		filters=64, kernel_size=(3, 3), padding="same"
	)(inputs)
'''


def create_discriminator(inputs):
	x = layers.Conv2D(
		filters=64, kernel_size=(3, 3), padding="same"
	)(inputs)
	x = layers.LeakyReLU()(x)

	x = layers.Conv2D(
		filters=64, kernel_size=(4, 4), padding="same", strides=(2, 2)
	)(x)
	x = layers.BatchNormalization()(x)
	x = layers.LeakyReLU()(x)

	x = layers.Conv2D(
		filters=128, kernel_size=(3, 3), padding="same"
	)(x)
	x = layers.BatchNormalization()(x)
	x = layers.LeakyReLU()(x)

	x = layers.Conv2D(
		filters=128, kernel_size=(4, 4), padding="same", strides=(2, 2)
	)(x)
	x = layers.BatchNormalization()(x)
	x = layers.LeakyReLU()(x)

	x = layers.Conv2D(
		filters=256, kernel_size=(3, 3), padding="same"
	)(x)
	x = layers.BatchNormalization()(x)
	x = layers.LeakyReLU()(x)

	x = layers.Conv2D(
		filters=256, kernel_size=(4, 4), padding="same", strides=(2, 2)
	)(x)
	x = layers.BatchNormalization()(x)
	x = layers.LeakyReLU()(x)

	x = layers.Conv2D(
		filters=512, kernel_size=(3, 3), padding="same"
	)(x)
	x = layers.BatchNormalization()(x)
	x = layers.LeakyReLU()(x)

	x = layers.Conv2D(
		filters=512, kernel_size=(4, 4), padding="same", strides=(2, 2)
	)(x)
	x = layers.BatchNormalization()(x)
	x = layers.LeakyReLU()(x)

	x = layers.Conv2D(
		filters=512, kernel_size=(3, 3), padding="same"
	)(x)
	x = layers.BatchNormalization()(x)
	x = layers.LeakyReLU()(x)

	x = layers.Conv2D(
		filters=512, kernel_size=(4, 4), padding="same", strides=(2, 2)
	)(x)
	x = layers.BatchNormalization()(x)
	x = layers.LeakyReLU()(x)

	x = layers.Flatten()(x)
	x = layers.Dense(100)(x)
	x = layers.LeakyReLU()(x)
	outputs = layers.Dense(1)(x)
	return Model(inputs, outputs, name="discriminator")


def create_generator(inputs, block_count=23, upscale_times=2):
	x = layers.Conv2D(
		filters=64, kernel_size=(3, 3), padding="same"
	)(inputs)
	residual = x

	for i in range(block_count):
		x = RRDBlock()(x)

	x = layers.Conv2D(
		filters=64, kernel_size=(3, 3), padding="same"
	)(x)
	x += residual

	for i in range(upscale_times):
		x = layers.Conv2D(
			filters=256, kernel_size=(3, 3), padding="same",
		)
		x = tf.depth_to_space(x, block_size=2)
		x = layers.LeakyReLU(alpha=0.2)(x)

	x = layers.Conv2D(
		filters=3, kernel_size=(3, 3), padding="same"
	)(x)
	x = layers.LeakyReLU(alpha=0.2)(x)
	x = layers.Conv2D(
		filters=3, kernel_size=(3, 3), padding="same"
	)(x)
	outputs = tf.identity(x)
	return Model(inputs, outputs, name="generator")