
## Building license plate number recognition with SageMaker built-in algorithm and Tensorflow BYOS (step-by-step)


Amazon SageMaker is a fully managed service with which developers and data scientists can build, train, and deploy ML models at any scale. To build your ML model with SageMaker, you can pick and use any of the built-in algorithms, you can use one of the popular deep learning frameworks with your own script (BYOS), or you can build your own algorithm with Docker container (BOYC).

In this blog post, you will learn how to use Amazon SageMaker built-in algorithms and TensorFlow BYOS to solve a real problem. Here the problem is to recognize characters of license plates from random images. The code example consists of 4 step self-study Hands on Labs that you can follow and you can download it from [here](https://github.com/mullue/lab-custom-model-anpr). The data processing code was inspired by [this link](https://github.com/matthewearl/deep-anpr). You I might want to visit our [developer guide](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html) in case you want to understand Amazon SageMaker in more detail. You can find more examples on [‘Object Detection algorithm on SageMaker’](https://aws.amazon.com/blogs/machine-learning/object-detection-algorithm-now-available-in-amazon-sagemaker/) and [‘Train and deploy Keras models with Tensorflow and Apache MXNet on SageMaker’](https://aws.amazon.com/blogs/machine-learning/train-and-deploy-keras-models-with-tensorflow-and-apache-mxnet-on-amazon-sagemaker/) posts.

<br /><br />

### Problem and project definition

We will divide our problem into two parts: the problem of detecting the area of the license plate from images and the problem of recognizing characters from license plate area. This choice can be an example of whether to solve the ML problem with end-to-end approach or subproblem separation.
The diagram below shows an example of approaches that prepare a machine learning project. Many commercial companies that deal with license plate recognition usually follow the subproblem separation approach like in the first picture. They detect license plate area first, and detect character’s image areas next, and then finally recognize the characters from each character's image areas. And, many end-to-end approaches are being experimented with like in the second picture. In our example, we will take a moderate approach like in the third picture.

<img src='imgs/ml_projects.png' stype='width:600px;'/>  

We will use the Australian license plate as test data. Each country's license plate has a limited number of characters. In Korea, the following 81 characters are used. You can generate random 7 character sequences (6 numbers and 1 Australian character at 3rd position) as test data by composing these characters. (source: [Wikipedia](https://ko.wikipedia.org/wiki/%EB%8C%80%ED%95%9C%EB%AF%BC%EA%B5%AD%EC%9D%98_%EC%B0%A8%EB%9F%89_%EB%B2%88%ED%98%B8%ED%8C%90))
```python
NUMS =['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
CHARS=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
SPACE=[' ']
JOIN =NUMS + CHARS + SPACE
```
As you have noticed, it is just a list of possible characters and we will use the index of this list as our training and inference. This means you can generalize our problem to the problem of other country's license plate character detection as well as the problem of reading the product serial numbers or signboards. For example, Japanese license plates can have the following characters with numbers according to Wikipedia. (source: [Wikipedia](https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_Australia))
```python
CHARS=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

```
<br />

We have understood a business problem. Now let’s solve this. In your [Jupyter notebook](https://docs.aws.amazon.com/sagemaker/latest/dg/nbi.html), copy the [source code](https://github.com/mullue/lab-custom-model-anpr) that we provide and follow the Lab0 through Lab3. 

<br />

### Preparing labeled data

In Lab 0, you will generate synthesized images with labels for your ML training. Image synthesis is a frequently used technique for data augmentation when you do not have enough data. (Even though they are less effective than actual data.)
To create images for the Lab, simply execute the following code in the provided ipynb cell.

```python
%run gen-w-bbx.py {num_of_images}
```

You will generate 2 kinds of labeled data pairs: raw image with its label consisting of license plate area position and cropped image with its label consisting of character index.
  
<img src='imgs/annotation.png' stype='width:600px;'/>  
  
<br /><br />
If you are wondering about the image synthesis, the code below is the secret. (It's in the gen-w-bbx.py file.)
```python
out = plate * plate_mask + bg * (1.0 - plate_mask)
```

Here, 'plate' is the random sequence of characters. 'plate_mask' is the position of the license plate, which you will add or subtract from plate and background images. In the next diagram, the first term 'plate * plate_mask' is the element-wise multiplication of the plate number image and plate_mask, and the second term 'bg * (1.0 - plate_mask)' will be the element-wise multiplication of background image and inverted mask. And then, you can get a final image by simply adding these two terms. (Lab uses Grayscale for the purpose of simplification.)
  
<img src='imgs/synthesis.png' stype='width:600px;'/>  
  
<br /><br />
  
### License plate area detection with SageMaker Object Detection algorithm

In Lab 1, you will develop your custom Object Detection model to detect the area of the license plate with Amazon SageMaker built-in Object Detection algorithm. (https://docs.aws.amazon.com/sagemaker/latest/dg/object-detection.html) You can just follow the guide of ipynb of Lab to upload files into S3, run your training job, deploy the trained model, and lastly test the inference from your deployed model.

One important concept you need to remember that in this Lab is Transfer Learning. As you use built-in algorithm of Amazon SageMaker you can turn on Transfer Learning by setting the ‘use_pretrained_model’ hyperparameter as 1(true), which enables you to leverage the pre-trained weights of CNN architecture. This allows you to obtain a high-quality model with a relatively small number of images. (The Lab uses 10,000 images as default but it also works well with 1,000 images.)


```python
od_model.set_hyperparameters(base_network='resnet-50',
                             use_pretrained_model=1,
                             num_classes=1,
                             mini_batch_size=32,
                             epochs=10,
                             learning_rate=0.001,
                             ...)
```
After the 10~20 minutes training, the training will end and you can see the result like below diagram. From the result list, the first number 0 signifies a label which means license plate in our example. The second 0.99... is the confidence for the detected object(license plate). The consecutive numbers from the third to the last mean relative x, y coordinates in the image, the width of detected area, and the height of the detected area, respectively.  

  
{'prediction': [[0.0, 0.9999839067459106, 0.1715950071811676, 0.27236270904541016, 0.808781623840332, 0.7239940166473389]]}  
<img src='imgs/od_sample1.png' stype='width:200px;'/>  
{'prediction': [[0.0, 0.9999842643737793, 0.20243453979492188, 0.3618628978729248, 0.8014888763427734, 0.6346850991249084]]}  
<img src='imgs/od_sample2.png' stype='width:200px;'/>  
{'prediction': [[0.0, 0.9999804496765137, 0.14474740624427795, 0.230726957321167, 0.8229358196258545, 0.7649730443954468]]}  
<img src='imgs/od_sample3.png' stype='width:200px;'/>   

<br />

### Preparing Tensorflow script with Keras

In Lab 2, you will write your own custom CNN architecture with TensorFlow and run it quickly to check if the code is grammatically correct. It will have 128 x 64 input layer, 3 Convolutional layers with Max Pooling and Batch Normalization, 1 Flatten layer right before the output layer, and finally 7 output layers for each character of license plate number. We will reshape input images to 128 x 64. The last output nodes express the probability of each classification among 81 characters.

You will see the summary of the architecture easily thanks to Keras as below.


```python
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 128, 64, 1)   0                                            
__________________________________________________________________________________________________
zero_padding2d (ZeroPadding2D)  (None, 132, 68, 1)   0           input_1[0][0]                    
__________________________________________________________________________________________________
conv0 (Conv2D)                  (None, 128, 64, 48)  1248        zero_padding2d[0][0]             
__________________________________________________________________________________________________
bn0 (BatchNormalization)        (None, 128, 64, 48)  192         conv0[0][0]                      
__________________________________________________________________________________________________
activation (Activation)         (None, 128, 64, 48)  0           bn0[0][0]                        
__________________________________________________________________________________________________
max_pool0 (MaxPooling2D)        (None, 64, 32, 48)   0           activation[0][0]                 
__________________________________________________________________________________________________
conv1 (Conv2D)                  (None, 60, 28, 64)   76864       max_pool0[0][0]                  
__________________________________________________________________________________________________
bn1 (BatchNormalization)        (None, 60, 28, 64)   256         conv1[0][0]                      
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 60, 28, 64)   0           bn1[0][0]                        
__________________________________________________________________________________________________
max_pool1 (MaxPooling2D)        (None, 30, 14, 64)   0           activation_1[0][0]               
__________________________________________________________________________________________________
conv2 (Conv2D)                  (None, 26, 10, 64)   102464      max_pool1[0][0]                  
__________________________________________________________________________________________________
bn2 (BatchNormalization)        (None, 26, 10, 64)   256         conv2[0][0]                      
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 26, 10, 64)   0           bn2[0][0]                        
__________________________________________________________________________________________________
max_pool2 (MaxPooling2D)        (None, 13, 5, 64)    0           activation_2[0][0]               
__________________________________________________________________________________________________
flatten (Flatten)               (None, 4160)         0           max_pool2[0][0]                  
__________________________________________________________________________________________________
d1 (Dense)                      (None, 81)           337041      flatten[0][0]                    
__________________________________________________________________________________________________
d2 (Dense)                      (None, 81)           337041      flatten[0][0]                    
__________________________________________________________________________________________________
d3 (Dense)                      (None, 81)           337041      flatten[0][0]                    
__________________________________________________________________________________________________
d4 (Dense)                      (None, 81)           337041      flatten[0][0]                    
__________________________________________________________________________________________________
d5 (Dense)                      (None, 81)           337041      flatten[0][0]                    
__________________________________________________________________________________________________
d6 (Dense)                      (None, 81)           337041      flatten[0][0]                    
__________________________________________________________________________________________________
d7 (Dense)                      (None, 81)           337041      flatten[0][0]                    
==================================================================================================
Total params: 2,540,567
Trainable params: 2,540,215
Non-trainable params: 352
__________________________________________________________________________________________________
```
<br />

### Bring your own script to SageMaker Tensorflow Estimator

In Lab 3, you will see how to modify the code of Lab 3 to run it on Amazon SageMaker. For TensorFlow versions 1.11 and later, the Amazon SageMaker Python SDK supports script mode training scripts. With this mode, you can write BYOS in much the same way as you would in an existing environment.

The most important modification to run the code on Amazon SageMaker is to match the input/output channels of the data. In terms of input channels, Amazon SageMaker training job runs on Docker Container and it assumes that the training data is on the S3.

The following picture illustrates the process of the data mapping that occurs in the Amazon SageMaker. At first, you upload your data into S3. Next, in your Python notebook, you will pass those S3 path into your training job as a parameter of JSON key-value format. And then Amazon SageMaker will copy the S3 data into the '/opt/ml/input/data/{channel}/' folder in your Docker Container and pass the paths as SM_CHANNEL_{channel} parameters. You should refer to those folders by using SM_CHANNEL_{channel} parameters or using those paths explicitly in your BYOS.(Look at the orange channel names in the picture below).


<img src='imgs/sm_data_path.png' stype='width:600px;'/>  
<br />
  
One more thing to understand is the control of hyperparameters. You may want to control hyperparameters like 'learning rate', 'number of epochs', etc. externally. Refer to the below picture. When you initiate your training job in your Jupyter notebook, hyperparameters will be passed as arguments of the Python run command-line script like the code in the middle of the picture. (You can find this command from the log of your training job in Lab3.)



<img src='imgs/sm_parameters.png' stype='width:600px;'/>  
<br />


You may refer to below resources for more information regarding Tensorflow script mode.

* https://aws.amazon.com/blogs/machine-learning/using-tensorflow-eager-execution-with-amazon-sagemaker-script-mode/
* https://docs.aws.amazon.com/sagemaker/latest/dg/tf.html
* https://github.com/aws-samples/amazon-sagemaker-script-mode


When you finish Lab3, you will see the result like below:

<img src='imgs/result1.png' stype='width:100px;'/>  
['3', '0', '호', '7', '3', '9', '3']  
<img src='imgs/result2.png' stype='width:100px;'/>  
['2', '3', '저', '9', '7', '2', '6']  
<img src='imgs/result3.png' stype='width:100px;'/>  
['0', '5', '느', '4', '9', '4', '6']  
<br />

You may find some mistakes like third result above. (it recognized Australian character '노' as '느'.) It is a natural at the first stage of the ML project. You may add more synthesized data or real data, change the internal architecture of CNN, or break the problem to 3 sub problems (Finding character areas and classifying the character), etc. You will repeat these experiments until you get the desired target quality.

In our case, adding more training data would be the first improvement we can try. Below you’ll find the result trained with 100,00 images, and you’ll notice steady improvement of accuracy.
  
<img src='imgs/finalresult.png' stype='width:600px;'/>  
<br />

### Going further

Now you can create your own custom Object Detection and custom TensorFlow CNN model with Amazon SageMaker if you have labeled data for the objects you want to track. Why don’t you extend the applications of this lab? For example, you may develop your own serial number detector of your product or your own custom pet detector from your mobile photo book like below illustration. 

<img src='imgs/further.png' stype='width:600px;'/>  

You might be worried about getting labeled data. But here we have [Amazon SageMaker Ground Truth](https://aws.amazon.com/ko/sagemaker/groundtruth/) to help you build highly accurate training datasets. You can find more examples of preparing training datasets from [here](https://aws.amazon.com/ko/blogs/machine-learning/tag/amazon-sagemaker-ground-truth/). 

