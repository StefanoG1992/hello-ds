# Hello ds

The "Hello world" of a 2020s developer is an image classification project.

In this project we take the famous [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
and build a convolutional neural network to predict one of its 10 classes.

The CNN is built using pytorch lightning, a framework to ease pytorch training, validation steps.
Data science testing is tracked via (local) mlflow.

#### About the NN
As we are using a local Mac to perform the testing, the net is not *so* deep, as it contains ~19.8K parameters.
As one can see in [Kaggle](https://www.kaggle.com/competitions/cifar-10/discussion), any pre-trained ResNet
or transfer-learning built algo can easily reach 99% accuracy on CIFAR10.

Given our stack capabilities, we preferred to focus more on the software engineering side of this project,
trying to build a deep learning setup(pytorch + lightning + mlflow) that can be re-used in multiple projects.

Nevertheless, it has to be noted that our small CNN (containing only 0.1% of ResNet parameters and only
a couple regularization techniques) already scores a ~55-60% accuracy, 5-6x times the accuracy of a random
predictor.
