# Description
The dataset was taken from kaggle https://www.kaggle.com/datasets/dansbecker/cityscapes-image-pairs/code. It contains city scapes images, belonging to 19 different classes. The goal is to train several segmentation models, including FCN, Unet, PSPnet.

# Data preparation
Train and valid data have 2975 and 500 images correspondingly. All images have format 256 x 512 per 3 RGB channels. Input and output images are placed side by side, so left half (256 x 256) is input image, and right half is output.
The ground truth images are used in format with one channel, where each pixel contains value from 0 to 18, meaning its class number. 

Since pixels in ground truth images are corrupted (there have to be 19 unique colors, but in fact there is more), colormap of true pixel colors for classes was used to calculate Euclidian distance from each pixel to each colormap color. Then pixel color was replaced with the one with less Euclidian distance from colormap. The colormap itself is taken from https://github.com/tensorflow/models/blob/17e923da9e8caba5dfbd58846ce75962206ffa64/research/deeplab/utils/get_dataset_colormap.py#L207.

# Models structure
## FCN
FCN model consists of encoder and decoder parts. Encoder includes 5 convolutional cells, where each cell is convolutinal layer with leaky relu activation followed by maxpool. Each cell reduces image linear sizes by 2, so the output feature map has shape 8 x 8. Decoder uses 5 transposed convolutional layers, and each increases image linear size by 2. So, each transposed convolution has corresponding convolutional cell from endoder, meaning that input image shape of first one is equal to output image shape of second one. Each transposed convolution layer, except last one, has leaky relu activation.

Summarized:
- (convolutional_layer + leaky_relu + maxpool) x 5
- (transposed_convolutional_layer + leaky_relu) x 4
- transposed_convolutional_layer

### Result metric
Pixel wised accuracy was used to evaluate model performance. The metric value on train data is 72% and on valid data is 70%. Note that valid dataset was not used for any kind of model optimization.

### Considerations
1. **Data preparation**. Augmentation technique, like mirroring, can be applied to every image.
2. **Weights initialization**. Using ReLU as activation function implies He inititalization for better data and gradient flow.
3. **Model architecture**. The predicted image consists of much simpler shapes than ground truth image. It may indicate that model failed to learn complex shapes, which may be potentially solved with adding more convolutional layers. Also, number of filters in existing layers can be increased.
4. **Regularization**. Model performance can be increased with adding Batch Normalization layers, which will enable model to learn layers output distribution, thus accelerating model convergence, and solve potential overfitting problem.
5. **Hyperparameters optimization**. No hyperparameters optimized at current step.
