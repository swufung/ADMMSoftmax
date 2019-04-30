# ADMMSoftmax

This repo contains a Matlab implementation of the ADMM-Softmax technique outlined in this [paper](https://arxiv.org/abs/1901.09450)

To run the experiments, we have six driver codes (three for MNIST and three for CIFAR-10):

  1) admmMNISTDriver.m
  2) newtonMNISTDriver.m
  3) sgdMNISTDriver.m
  
  4) admmTransferLearningCIFAR10.m
  5) newtonTransferLearningCIFAR10.m
  6) sgdTransferLearningCIFAR10.m

Simply run them in MATLAB.

You will also need to add the paths to where your Meganet folder is located.

If the user does not have the MNIST and CIFAR10 dataset, Meganet will give them the option to download the files automatically once the “setupCIFAR10” or “setupMNIST” function is called in the driver.


# Dependencies

The codes here rely on the following:
  1) The forked Meganet package found [here](https://github.com/samywu/Meganet.m)
	2) MATLAB 2018b which contains MATLAB’s deep neural network package (this is for the transfer learning component in the CIFAR-10 experiments).

# Acknowledgments

This material is supported by the U.S. National Science
Foundation (NSF) through awards DMS 1522599 and DMS
1751636.
