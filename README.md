# CSC591 Capstone Project - Yelp Restaurant Photo Classification

### Team Members:
  - Amit Watve (awatve@ncsu.edu)
  - Omkar Acharya (oachary@ncsu.edu)
  - Akshay Arlikatti (aarlika@ncsu.edu)

### Description:
In this project, we build a model that automatically tags restaurants with multiple labels using a dataset of user-submitted photos. Currently, restaurant labels are manually selected by Yelp users when they submit a review. Selecting the labels is optional, leaving some restaurants un- or only partially-categorized. 
In an age of food selfies and photo-centric social storytelling, it may be no surprise to hear that Yelp's users upload an enormous amount of photos every day alongside their written reviews. 

### You must have ..
* [Numpy] - For handling the datasets   (```pip install numpy```)
* [Pandas] - For handling the datasets  (```pip install pandas```)
* [Scikit Learn] - To use classification algorithms like SVM    (```pip install -U scikit-learn```)
* [Python]

The following dependencies are only required if you wish to extract image and business features from scratch. But we have already done that for you, you just need to download them from the links provided below in the table. Make sure that you put these files in "features" directory.
* [H5Py] - To store the features extracted from CNN (```pip install h5py```)
* [Caffe] - To extract features from the images (Refer to the [link](http://caffe.berkeleyvision.org/))

### Folder description:
* ```code/``` - contains programs to extract features and perform the final classification.
* ```data/``` - contains training and testing images + metadata from Yelp dataset (We have already extracted and stored the features for east of project execution).
* ```features/``` - contains the extracted features from images and restaurants (For ease of project execution).
* ```models/``` - contains trained SVM model which can be used for future predictions without retraining (Will be generated automatically when ```classify.py``` is run for the first time; for ease of project execution, we have included this model as well).

### Dataset:
Again if you choose to extract image and business features from scratch, you will need this dataset. It is available [here](https://www.kaggle.com/c/yelp-restaurant-photo-classification/data). Dataset description is also available. Download and extract the files/folders in the "data" directory.

### For ease of project execution, we have already extracted the features and stored in the following files:

| Filename                    | Size    | Description                                                                                                                   | Command that was used for generation                                   |
|-----------------------------|---------|-------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------|
| train_features.h5           | 3.59 GB | Format: [PhotoId, ImageFeatures] This file contains ImageNet features of training dataset                                     | ```python extract_image_features_train.py```    |
| test_features.h5            | 18.2 GB | Format: [PhotoId, ImageFeatures] This file contains ImageNet features of test dataset                                         | ```python extract_image_features_test.py```     |
| [train_business_features.csv](https://drive.google.com/file/d/0B9_9jccGmC69Z1BQZEpURUNpQ28/view?usp=sharing) | 91.7 MB | Format: [BusinessId, BusinessFeatures, ClassLabels] This file contains features extracted for businesses in training dataset. These features are extracted using train_features.h5.  | ```python extract_business_features_train.py``` |
| [test_business_features.csv](https://drive.google.com/file/d/0B9_9jccGmC69cDNlRVMzUFJnRzQ/view?usp=sharing)  | 460 MB  | Format: [BusinessId, BusinessFeatures] This file contains features extracted for businesses in test dataset. These features are extracted using test_features.h5.                   | ```python extract_business_features_test.py```  |


### To perform final classification:
```sh
$ cd code
$ python classify.py
```

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)


   [Numpy]: <http://www.numpy.org/>
   [Pandas]: <http://pandas.pydata.org/>
   [Caffe]: <http://caffe.berkeleyvision.org/>
   [H5Py]: <http://www.h5py.org/>
   [Scikit Learn]: <http://scikit-learn.org/stable/>
   [Python]: <https://www.python.org/>
