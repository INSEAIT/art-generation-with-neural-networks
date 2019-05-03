# art-generation-with-neural-networks
A style transfer application that is able to apply a new style to an image while still preserving its original content.

## Implementation Details
This implementation of style transfer uses TensorFlow to train a style transfer network. It follows the same transformation network as described in the videos of deeplearning.ai from the course of convolutional neural networks on coursera.
Using the VGG19 pretrained model on ImageNet.
### Step1 :
So before trying the model, download the pretrained model from [here](https://www.kaggle.com/teksab/imagenetvggverydeep19mat#imagenet-vgg-verydeep-19.mat) and extract it to the folder ('pretrained-model').
### Step 2 : 
Add both your content image and style image to the folder ('images') and make sure to reshape them to 400x300 and rename them respectively to my_content and my_style or you can just edit the code to use the names you like : 
```
content_image = scipy.misc.imread("images/your_image_here.jpg")
style_image = scipy.misc.imread("images/your_style_here.jpg")
```
### Step 3 : 
in the line 69 , the number of itereration by default is 400 , you can increase the number and see if the perfoemance will increase too ! 
```
def model_nn(sess, input_image, num_iterations = 400):
```
### Attributions/Thanks 
- The project also borrowed some code from Anish's [Neural Style](https://github.com/anishathalye/neural-style/)
- Some readme/docs formatting was borrowed from longstorm [fast-style-transfer
](https://github.com/lengstrom/fast-style-transfer)

