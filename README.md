<h1>face-landmark-localization</h1>
<h2>Overview</h2>
This is a Keras implementation of a simple face landmark localization model. It is implemented based on adding the final regression layer to an infamous pretrained networks (provided by Keras).

<h2>Dataset</h2>
The dataset used here is MUCT face database consisting of 3,755 faces each with 76 hand labeled landmarks. The dataset can be found at: https://github.com/StephenMilborrow/muct. Note that here we resize the original images by a factor of 2 in both width and height (from 640x480 to 320x240) so that my GPU memory is sufficient for the training (1080ti 11GB). The target data (ground truth points) is then also needed to be scaled down by a factor of 2 as well, i.e. (128, 135) to (64, 67.5).

<h2>Model</h2>
<h3>Architecture</h3>
The neural network architecture here is ResNet50 with pretrained weights from ImageNet. A global average pooling is added to the net to reduce dimension and the last regression layer with 152 outputs (for each of the 76 landmark points) and with Sigmoid activation function is added.

<h3>Loss function</h3>
The loss function used in training the mean of Euclidean distance between the ground truth points and the predicted points for each mini-batch. Points which could not be located by human is indicated by coordinate of (0, 0) in the dataset. These points are excluded in the calculation of the loss function.

<h3>Training</h3>
Dataset is divided into 70% (2628 images) for training and 30% (1127 images) for testing. The ground truth points are first normalized by the width and height of the image before feeding to the network. For the results in the next section, the training parameters are as follows:

  * Batch size -- 16
  * Optimizer -- Adam
  * Learning rate -- 0.001
  * number of epochs -- 8
  
The pretrained weights can be found at: https://drive.google.com/file/d/1aEZX1TzmLlsHuN6vLe3neRxlp9_f6486/view?usp=sharing

<h2>Result</h2>
The results by testing with this dataset could not be shown due to privacy agreement. Results on other dataset will be added later on.
