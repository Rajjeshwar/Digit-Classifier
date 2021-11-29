# Digit-Classifier
Digit classifier to predict handwritten digits

To understand what a convolution is please refer to theory.md

Let's understand what convolutional neural networks do, 

CNNs are made up of neurons that have learnable weights and biases. Each neuron receives some inputs, performs a dot product and optionally follows it with a non-linearity. The whole network still expresses a single differentiable score function: from the raw image pixels on one end to class scores at the other. And they still have a loss function (e.g. SVM/Softmax) on the last (fully-connected) layer and all the tips/tricks we developed for learning regular Neural Networks still apply.

So what changes? ConvNet architectures make the explicit assumption that the inputs are images, which allows us to encode certain properties into the architecture. These then make the forward function more efficient to implement and vastly reduce the amount of parameters in the network.

CS231n does a wonderful job of breaking down the components of a CNN as shown below.

![image](https://user-images.githubusercontent.com/80246631/142727278-8ccccadf-4ba0-44e1-9475-2e7fdee2eec6.png)


### Convolution Layer:

![image](https://user-images.githubusercontent.com/80246631/142725232-c69c5d93-bd78-4dab-ae2b-0442a9c9043e.png)

1. Accepts a volume of size W1×H1×D1
2. Requires four hyperparameters:
3. Number of filters K,
4. their spatial extent F,
5. the stride S,
6. the amount of zero padding P.
7. Produces a volume of size W2×H2×D2 where:
8. W2=(W1−F+2P)/S+1
9. H2=(H1−F+2P)/S+1 (i.e. width and height are computed equally by symmetry)
10. D2=K
11. With parameter sharing, it introduces F⋅F⋅D1 weights per filter, for a total of (F⋅F⋅D1)⋅K weights and K biases.
12. In the output volume, the d-th depth slice (of size W2×H2) is the result of performing a valid convolution of the d-th filter over the input volume with a stride of S, and then offset by d-th bias.

### Pooling Layer:

![image](https://user-images.githubusercontent.com/80246631/142725195-434400fe-c58d-4b4f-888c-d685777cd65f.png)

1. Accepts a volume of size W1×H1×D1
2. Requires two hyperparameters:
3. their spatial extent F,
4. the stride S,
5. Produces a volume of size W2×H2×D2 where:
6. W2=(W1−F)/S+1
7. H2=(H1−F)/S+1
8. D2=D1
9. Introduces zero parameters since it computes a fixed function of the input
10. For Pooling layers, it is not common to pad the input using zero-padding.
11. It is worth noting that there are only two commonly seen variations of the max pooling layer found in practice: A pooling layer with F=3,S=2 (also called overlapping pooling), and more commonly F=2,S=2. Pooling sizes with larger receptive fields are too destructive.


## Design decisions:

1. We have used 3 `conv2d` layers, ie- we have convolved the input images in three different layers. Although going deeper doesn't necessarily entail that the network will perform better (cue vanishing gradient!), in this case the input data and the number of parameters to be learnt are quite small so it does not lead to such a problem.
2. We use padding to keep the dimensionality of the output of every convolution layer the same, since the size of the images are alreaady quite small. We do this by using the `padding="same"` parameter.
3. Adam optimizer is used as a replacement for alternatives like SGD or RMSProp etc. This is done to reduce training time.
4. A max pool layer is used for sharp feature extraction. Since the images are greyscale and have sharp edges the max pool layers really help a network extract and 'understand' all the features.
5. Since we are already using a pooling layer for downsampling, the strides for kernel filter has been left at the default value of 1.
6. We opt for a ReLU non linearity as the activation function for all our layers except the output layer. Default values of learning rate and beta1 and beta2 are used. 
7. Our final layer is a dense layer of 10 hidden units (number of output labels needed for MNIST digit dataset). Here, the softmax activation is used to output a probability score for each of the 10 labels.

## Evaluation: 

CE loss is used since this is a probabilisitic problem and not a regression problem. Since our output labels are one hot encoded vectors of length 10 we use `categorical crossentropy` to get our output.

 ```
 train accuracy: 0.9960
 
 ```
 
 The output of this code tested against the test set achieved an accuracy of 0.98375 when submitted
 
 ## Install Requirements: 
 
The following were used for making this program-

1. Tensorflow
2. sklearn
3. numpy
4. pandas
5. os module
6. unittest
7. Kaggle
 
 ```
 pip install -r requirements.txt
 ```
 
 The following link provides a good walkthrough to setup tensorflow:
 
  ```
https://towardsdatascience.com/tensorflow-gpu-installation-made-easy-use-conda-instead-of-pip-52e5249374bc
 ```
  ## Usage:
 
 ```
 conda install -c conda-forge jupyterlab
 jupyter-lab
 #open Digit_classifier_main.ipynb
 ```
 
 ## Format code to PEP-8 standards (Important for contributing to the repo): 
 
 This repository is strictly based on *PEP-8* standards. To assert PEP-8 standards after editing your own code, use the following: 
 
 ```
 black Digit_classifier.py
 ```
## Dataset download 

Running the following cell downloads the dataset using the Kaggle api. 

```
get_ipython().system("kaggle competitions download -c digit-recognizer")
```
### Reference: 

1. https://cs231n.github.io/convolutional-networks/
2. https://www.uksim.info/isms2016/CD/data/0665a174.pdf
3. https://towardsdatascience.com/what-is-stratified-cross-validation-in-machine-learning-8844f3e7ae8e
4. https://www.superdatascience.com/blogs/convolutional-neural-networks-cnn-softmax-crossentropy
