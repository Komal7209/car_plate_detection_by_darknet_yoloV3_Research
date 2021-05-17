# Car_plate_detection_by_darknet_yoloV3_R


## Tools could be used:
- Then I explored which language I could use: **MATLAB** or **python**
- Another take was on which library I need to choose **tensorflow** , **keras** , **scikit-learn** , **pytorch**
- How to image could be processed: **using opencv**
- Frameworks could be used: **darknet** , **darkflow(tensorflow version of darknet)** ,**LPRNet** , **Tesseract**
- if wanna go directly with an API then it's : **GoogleVision**

Final product is made using **darknet** framework, **yolov3** weights and used **google colab** so that it could use cloud's hardware for processing and cloud's storage for storing the dataset using it as **IaaS** and **Sar**

## Some more explanation related to tools

### Core Framework and Tools

- **Python** is a very popular high-level programming language that is great for data science. Its ease of use and wide support within popular machine learning platforms, coupled with a large catalog of ML libraries, has made it a leader in this space.
- **Pandas** is an open-source Python library designed for analyzing and manipulating data. It is particularly good for working with tabular data and time-series data.
- **NumPy**, like Pandas, is a Python library. NumPy provides support for large, multi-dimensional arrays of data, and has many high-level mathematical functions that can be used to perform operations on these arrays.

### Machine Learning and Deep Learning
 
 - **Scikit-Learn** is a Python library designed specifically for machine learning. It is designed to be integrated with other scientific and data-analysis libraries, such as **NumPy**, **SciPy**, and **matplotlib**.
- **Apache Spark** is an open-source analytics engine that is designed for cluster-computing and that is often used for large-scale data processing and big data.
- **TensorFlow** is a free, open-source software library for machine learning built by Google Brain.
- **Keras** is a Python deep-learning library. It provide an Application Programming Interface (API) that can be used to interface with other libraries, such as TensorFlow, in order to program neural networks. Keras is designed for rapid development and experimentation.
- **PyTorch** is an open source library for machine learning, developed in large part by Facebook's AI Research lab. It is known for being comparatively easy to use, especially for developers already familiar with Python and a Pythonic code style.

## YoloV3 functioning:

"Selects box with highest probability from all boxed objects."

### Key Summary
* You Look Only Once (Yolo) Implemented in Tensorflow
* K-means clustering across short rolling windows to group similar objects across frames
* Predict if object is similar between frames

### Yolo Basics
* Single Feedforward network
* Yolo reframe object detection as single regression problem
* Images -> Pixels -> Bounding Box -> Probabilities
* Divides image into S X S Grid
* If Object centre falls in the Grid then grid cell is responsible for detecting the object
* Predict bounding boxes and class probabilities

### Yolo Implementation
* 32 layer Deep CNN
* Input Image resized into 448 x 448
* Yolo Network Output = S x S x (B*5 + C) tensor of predictions ( 7 x 7 x 30 )
* S - Numbers of rows and columns in which we divide the image
* B - Number of objects that can be predicted in given box
* C - Number of classes
* 5 - Terms account for x-axis grid offset, y-axis grid offset, width, height and confidence in each grid cell
* Dataset - Pascal VOC Dataset

### K-Means Extension to Yolo
* K-means clustering across images within the short rolling window to group similar objects across frames
* Define Distance between two images I1, I2 given dimensions x,y and color channels c
* Works well when images are similar 

# **Prerequisites**

## Software Requirements:
  * Google colab notebook or jupyter Notebook
  * Python 3.8
  * OpenCV
  
## Hardware Requirements: 
  * NVIDIA GPU 940 MX and above
  * 12 GB Ram and above


## **Google Colab Notebooks**

## For environment set up( installation notebook):
https://colab.research.google.com/drive/1A0QJcqFwNJV0_y7TAE4ZSeB9VVO_3ceb?usp=sharing

</br>

## For Model training (training notebook) :
https://colab.research.google.com/drive/1jsVbw4L2FKPcB9q4-YMvR6ZEKoQjvaGw?usp=sharing

</br>

## For using the model over example image( demo notebook):
https://colab.research.google.com/drive/1WnPkQEaDg93u6nMNywZNgZwpIMdkbbMR?usp=sharing

</br>

## **Resources used**

## Link for dataset:
- http://www.zemris.fer.hr/projects/LicensePlates/english/  (used here) </br>
- https://datasetsearch.research.google.com/

## link for labelling dataset:

- https://github.com/tzutalin/labelImg (used here) </br>
- https://github.com/EscVM/OIDv4_ToolKit </br>
- https://github.com/theAIGuysCode/OIDv4_ToolKit </br>

- *Have used instructions as per above two repos only. Later one only provide with a command at top to provide final output of dataset.*
- *Further dataset has been directly uploaded to google drive for training the model*
- *Got error with AWS Client while fetching images as per these repos and had solved that by upgrading.*

## References
- https://colab.research.google.com/drive/1lTGZsfMaGUpBG4inDIQwIJVW476ibXk_#scrollTo=jAN2TNZ007c_ </br>
- https://colab.research.google.com/drive/1Mh2HP_Mfxoao6qNFbhfV3u28tG8jAVGk#scrollTo=VHw00Cro8ONr  </br>

## File Repo Reference
- https://github.com/theAIGuysCode/YOLOv3-Cloud-Tutorial/tree/master/yolov3   </br>
- https://github.com/theAIGuysCode/YOLOv3-Cloud-Tutorial                       </br>
- https://blog.francium.tech/custom-object-training-and-detection-with-yolov3-darknet-and-opencv-41542f2ff44e


## Presentation Website
https://komal7209.github.io/project-presentation-website/
