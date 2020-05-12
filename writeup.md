# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

This report constitutes the write up for the Traffic Sign Classification project. It should be read in collaboration with the HTML notebook output provided in the project workspace. The code is also available on github at [project code](https://github.com/mikec-w/Udacity-CarND-Traffic-Sign-Classifier)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Please refer to the HTML notebook output for the visualisation. Here the first example of each unique traffic sign in the training set is displayed along with its index and description. It is sorted in the order of the training set and while only displaying a limited number of images it shows some interesting points of note. Particularly range of brightness and contrasts seen. 

In addition to the plot of the road signs, a histogram was provided that showed the distribution of the different type of road signs in training, testing and validation set. This does suggest that all sets are not particularly balanced with a much higher frequency of examples of signs from the lower indices than the higher ones. While this may be representative of the actual frequency these road signs may appear in the real world, a more balanced set may be beneficial in making the resulting network more robust.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As all the images of the training set, test set and validation set have already been sized to 32x32, this step was not necessary. It would make sense to add it to the preprocessing function however to ensure that it is more robust against any image provided. 

While the project description suggests the conversion to grayscale, this seems counter-intuitive as road signs use colour to differeniate themselves. Accordingly this step was not taken but the image colour space was normalised as recommended to improve final network fit.

Finally the training set was shuffled to ensure no order based features persisting.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The final model consisted of the following layer based primarily on the LeNet labs tutorial. The dimensions have been adjusted to suit this application and the addition of dropout has been used to try and ensure network robustness.


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					| With dropout 									|
| Max pooling	      	| 2x2 stride, outputs 14x14x6 with dropout		|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10x16 	|
| RELU					| With dropout 									|
| Max pooling	      	| 2x2 stride, outputs 5x5x16 with dropout		|
| Flatten				| output 1x400 									|
| Fully connected		| output 120   									|
| RELU					| With dropout 									|
| Fully connected		| output 84   									|
| RELU					| With dropout 									|
| Fully connected		| output 43 									|

The model returned logits, a softmax function was then applied to the output prediction
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Training the model was performed using the AdamOptimiser attempting to minimize the average cross entropy. For the hyperparameters, the batch_size was left at 128 as this seemed to give reasonable performance using the Udacity workspace. It is possible that with more information on the GPU implementation this could be adjusted to improve hardware usage but was not necessary in this case. 

The training rate employed was 0.001 and variation on this only seemed to reduce the overall accuracy of the final network. 

The EPOCHS hyper parameter however was increased to 50 as in some of the attempts to train the model, the accuracy did not seem to plateau to any great extent within fewer EPOCHS. In the final training (shown in the HTML output) this did not seem to be too much of an issue with the accuracy stabilising in the low 20's, but in previous examples it has required a few more runs through.

Finally the dropout rate was set at 85%. Any lower and the accuracy started to tail off while any higher risked reducing the robustness of the model.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.99%
* validation set accuracy of 95.1% 
* test set accuracy of 94.8%

The architecture chosen for the classification of traffic signs was based on the original LeNet architecture presented in the course material. This was chosen primarily on the advice of the course but is also a logic option. The original purpose of LeNet was to classify images (hand written digits) which is functionally the requirement in this case.

Some modifications were required, in particular adjusting the input and output dimensionality to cope with a 32x32x3 image input and providing an output for all 43 classes of road sign. Given the increased number of output classes there is a question as to whether the final three fully connected layers are necessary to reduce the dimensionality from 400 to 120 to 84 to 43. This is something that could be investigated further but given that the layers are essentially seperating a hiearchy of features it is not expected to show significant gains. Additionally there is a question as to whether an extra layer would improve this as the original architecture was used on grayscale images whereas in this case colour is an addition to the feature hiearchy that may benefit from its own part of the model. The answer to that question though is hard to determine intuitively as the self-learning nature of the network means that it is not a simple task to specify the purpose of an hidden layer. 

In an attempt to improve the robustness of the architecture, dropout stages were liberally scattered throughout the model all with the same probability of a neuron being kept. Here a more disjunct approach could have been taken to determine where best to place such stages but that was beyond the scope of this project.

Once the model seemed appropriate for the training set, various attempts were made to train the model with different hyperparameters. The learning rate, EPOCHS and keep probability were all adjusted until the final accuracy level approached a good balance. A technique similar to gradient descent was applied to the hyperparameters with reasonable steps in each direction from a point to determine the gradient of the accuracy and quickly come to a reasonable point. In the end the most significant change seemed to be to increase the EPOCHS until the accuracy had seemed to stabilise. 

In the end the balance between the training set, validation set and test set seemed reasonable to suggest the model exhibited neither signs of overfitting or underfitting while giving a high level of accuracy.

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Step 3 of the HTML output shows five road signs taken from the internet. It was expected that the first sign, a left turn, might be tricky due to the blue background sky almost matching the sign itself. Similarly with the 5th image, the background colour and contract matched the sign itself.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Left Turn      		| Left Turn   									| 
| 70kph     			| 70kph 										|
| Priority 				| Priority										|
| Stop Sign     		| Stop Sign 					 				|
| No Entry Sign			| No Entry Sign      							|

The model was able to correctly identify all 5 of the traffic signs giving it a 100% accuracy. This compares favourably to the accuracy of the test set although there is still scope to improve the model futher as the sample of 5 images does not allow for much resolution in this accuracy. The validation set should give a more representative answer with its score of 94.8% which, while not bad at all, could be improved upon.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The final section of the HTML notebook output shows the softmax probabilities for each of the five test images and show that in each case the network was reasonable confident with the lowest probably of the prediction for the Turn Left sign at 99.79% confidence (this being the image initially suspected to be tricky due to the blue background). Interestingly the no entry sign expected to be slightly tricky returned with a  100% confidence level.

The top five soft max probabilities for thge left turn sign were: (the probabilities for the other four images are in the HTML notebook)

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 99.79%       			| Turn Left   									| 
| 0.11%    				| Ahead Only									|
| 0.07%					| Roundabout									|
| 0.04%	      			| Keep Right					 				|
| 0.00%				    | Turn Right   									|



