# Date Fruit Classification
<p align="center">
  <img src="https://user-images.githubusercontent.com/57125377/206633540-c821cdff-01db-4f79-9dc4-35af0e678742.jpg">
</p>

---
## Index

- 1 [Description of the problem](#1-description-of-the-problem)

- 2 [Instructions on how to run the project](#instructions-on-how-to-run-the-project)
-
---
## Structure of the repository

The repository conatins the next files and folderrs:

- `images`: folder with images to README.md

## 1. Description of the problem

<p align="justify">
A great number of fruits are grown around the world, each of which has various types. The factors that determine the type of fruit are the external appearance features such as color, length, diameter, and shape. The external appearance of the fruits is a major determinant of the fruit type. Determining the variety of fruits by looking at their external appearance may necessitate expertise, which is time-consuming and requires great effort. 
</p>

## 2. Objective

<p align="justify">
The aim of this project is to classify the types of date fruit<b>, that are Barhee, Deglet Nour, Sukkary, Rotab Mozafati, Ruthana, Safawi, and Sagaiz</b> by using <b>a deep neural network model</b>.
</p>

## 3. Data description

<b>Data set source:</b>
> https://www.muratkoklu.com/datasets/

<p align="justify">
In accordance with this purpose, 898 images of seven different date fruit types were obtained via the computer vision system (CVS). Through image processing techniques, a total of 34 features, including morphological features, shape, and color, were extracted from these images. 
</p>

## 4. Setting up the virtual environment

<p align="justify">
A virtual environment allows us to manage libraries or dependencies for different projects without having the version compatibility problem by creating isolated virtual environments for them. There are many environments managment systems for python such as conda, pipenv, venv, virtualenv and so on, but for this project I used pipenv. 
</p>

<p align="justify">
Next, I'll explain how to install pipenv, and create an environment for a python project.
Before starting , first we need to install pip, which is a package-management system to install python packages, run these codes in the console.
</p>

> For windows:

    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

    python get-pip.py
  
> For linux:

    sudo apt update

    sudo apt install python3-pip
  
Then after installing pip, we need to install pipenv
    
    pip install pipenv
    
<p align="justify">
Once here, it's necessary to clone this repository from Github to your computer, so create a folder in the path of your choice, and give it a name.
</p>

    mkdir ./file_name
    
    cd ./file_name
    
    git clone git@github.com:JesusAcuna/Date_Fruit_Classification_Keras.git

After that, we need to activate the virtual environment
  
    pipenv shell
    
And install the content of these file`Pipfile` and `Pipfile.lock`, these ones contain information about the libraries and dependencies I used.

    pipenv install
    
To exit the environment just type exit, when you are in the environment
  
    exit
    
For this project I used these libraries:
- flask          : to build the web service framework
- tflite-runtime : lite tensorflow library for prediction
- requests       : to make request to the web service 
- joblib         : to load the normalization object
- scikit-learn   : to apply the normalization transformation to our request

## 5. Importing data

<p align="justify">
We can download the data from the web : https://www.muratkoklu.com/datasets/vtdhnd06.php, this file is a zip file, so we need to make a request to that URL, save its content and extract all files it contains, the code below is the first part of the `Date_Fruit_Classification.ipynb` and allows you to download it to the current path. The archive we are interested in is Date_Fruit_Datasets.xlsx, which is an excel extension and this is the data that I'll work all the project, also this data is in the repository `Date_Fruit_Datasets`.
</p>

    # Importing necessary modules
    import requests, zipfile
    from io import BytesIO

    # Defining the zip file URL
    url = 'https://www.muratkoklu.com/datasets/vtdhnd06.php'

    # Downloading the file by sending the request to the URL
    req = requests.get(url)

    # Extracting the zip file contents
    zipfile= zipfile.ZipFile(BytesIO(req.content))
    zipfile.extractall('./') #Current directory

## 6. Notebook

Data preparation, data cleaning, EDA, feature importance analysis, model selection and parameter tuning was performed in `Date_Fruit_Classification.ipynb` 
 
### 6.1. Data preparation and data cleaning 

<p align="justify">
Data contains 898 examples,33 features, and a target variable of 7 classes in total explained in [Data description](#3-data-description), these features are external appearance features such as area, perimeter, shape factor, color and so on, check the notebook out for more information, the main characteristic is that they are all numerical features, and some are bigger values than other ones, that's why I applied a normalization with a mean equals to 0 and a standard deviation equals to  1
</p>


## Instructions on how to run the project



## References


## Contact 



