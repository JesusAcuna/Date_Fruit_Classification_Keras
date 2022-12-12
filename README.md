# Date Fruit Classification
<p align="center">
  <img src="https://user-images.githubusercontent.com/57125377/206633540-c821cdff-01db-4f79-9dc4-35af0e678742.jpg">
</p>

---
## Index

- 1.[Description of the problem](#1-description-of-the-problem)
- 2.[Objective](#2-objective)
- 3.[Data description](#3-data-description)
- 4.[Setting up the virtual environment](#4-setting-up-the-virtual-environment)
- 5.[Importing data](#5-importing-data)
- 6.[Notebook](#6-notebook)
  - 6.1.[Data preparation and data cleaning ](#61-data-preparation-and-data-cleaning) 
  - 6.2.[Exploratory Data Analysis (EDA)](#62-exploratory-data-analysis-eda)
  - 6.3.[Feature importance analysis](#63-feature-importance-analysis)
  - 6.4.[Model selection and parameter tuning](#64-model-selection-and-parameter-tuning)
- 7.[Instructions on how to run the project](#7-instructions-on-how-to-run-the-project)
- 8.[Locally deployment](#8-locally-deployment)
- 9.[Google Cloud deployment (GCP)](#9-google-cloud-deployment-gcp)
- 10.[References](#10-references)
---
## Structure of the repository

The repository contains the next files and folders:

- `150_Stability_045011`: directory of the stability with different test sizes of the 4 trained models 
- `4_Models_000123`: directory of model history and parameters
- `Date_Fruit_Datasets`: irectory of the data set
- `images`: directory with images to README.md
- `Best_Model_3.h5`: archive of best chosen model 
- `Best_Model_3.tflite`: archive with extension tensorflow lite of best chosen model
- `Date_Fruit_Classification.ipynb`: python notebook where the analysis and modeling is done
- `Dockerfile`: archive to containerize the project
- `Pipfile`: archive to save the dependencies and libraries of the environment
- `Pipfile.lock`: archive to save the cache of the environment
- `convert_to_tflite.py`: python script to convert a h5 file to tfile file
- `predict.py`: python script to make the web service with method 'POST' and upload the parameters of best model
- `predict_test.py`: python script to make a request locally
- `predict_test_cloud.py`: python script to make a request to Google Cloud Platform (GCP)
- `std_scaler.bin`: binary archive with the training normalization values 
- `train.py`: python script to train the model

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
A virtual environment allows us to manage libraries or dependencies for different projects without having version compatibility problem by creating isolated virtual environments for them. There are many environments managment systems for python such as conda, pipenv, venv, virtualenv and so on, but for this project I used pipenv. 
</p>

<p align="justify">
Next, I'll explain how to install pipenv, and create an environment for a python project.
Before starting , first we need to install pip, which is a package-management system to install python packages. Run these codes in the console.
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
    
And install the content of these files `Pipfile` and `Pipfile.lock`, these ones contain information about the libraries and dependencies I used.

    pipenv install
    
To exit the environment just type exit
  
    exit
    
For this project I used these libraries:
- flask          : to build the web service framework
- tflite-runtime : lite tensorflow library for prediction
- requests       : to make request to the web service 
- joblib         : to load the normalization object
- scikit-learn   : to apply the normalization transformation to our request
- waitress       :

## 5. Importing data

<p align="justify">
We can download the data from the web : https://www.muratkoklu.com/datasets/vtdhnd06.php, this file is a zip file, so we need to make a request to that URL, save its content and extract all the files it contains, the code below is the first part of the Date_Fruit_Classification.ipynb and allows you to download it to the current path. The archive we are interested in is `Date_Fruit_Datasets.xlsx` , which is an excel extension and this is the data that I'll work all the project.
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
<p align="justify">
Data preparation, data cleaning, EDA, feature importance analysis, model selection and parameter tuning was performed in Date_Fruit_Classification.ipynb
</p>
 
### 6.1. Data preparation and data cleaning 
<p align="justify">
The data contains 898 examples, 33 features, and a target variable of 7 classes, this was explained in point (3.Data description).These features are external appearance features such as area, perimeter, shape factor, color and so on, check the notebook out for more information. The dataframe doesn't contain missing values, and to train the model it's required to change the target variable from object to numerical like below.
</p>

<p align="center">  
['DOKOL': 0, 'SAFAVI': 1, 'ROTANA': 2, 'DEGLET': 3, 'SOGAY': 4, 'IRAQI': 5, 'BERHI': 6]
</p>  

<p align="justify"> 
The main characteristic is that they are all numerical features, and some are larger values than other ones, that's why I applied normalization with a mean equals to 0 and a standard deviation equals to  1. To do this part I used StandardScaler from sklearn.preprocessing to standarize all the features, then I saved the object using the `joblib` library with the name `std_scaler.bin`, this archive will be used later to make the predictions.
</p>

### 6.2. Exploratory Data Analysis (EDA)

<p align="justify"> 
From the image below, we can see that there are about 200 examples where the target variable is 'DOKOL' or 'SAFAVI'. Both classes along with 'ROTANA' are the three largest classes, on the other hand 'BERHI','DEGLET','IRAQI' and 'SOGAY' have examples lower than 100. Also, I did some analysis on some features like area distribution and I found out that three classes had almost the same distribution range, and could be replaced by one, you can analyze all the features if you would like, but the idea is the same.
</p>

<p align="center">
  <img src="https://github.com/JesusAcuna/Date_Fruit_Classification_Keras/blob/main/images/target_distribution.png">
</p>

### 6.3. Feature importance analysis

<p align="justify"> 
Since all the features are numerical I did a Pearson correlation coefficient analysis, which measures the linear relationship between two variables. This has to be done after normalization of the data, as seen in point (6.3. Data preparation and data cleaning), you can visualize it below 
</p>

<p align="center">
  <img src="https://github.com/JesusAcuna/Date_Fruit_Classification_Keras/blob/main/images/correlation.png">
</p>

<p align="justify"> 
The results show that there are 13 features that have values below 1e-1, that means these 13 features are not correlated at all. so they could be deleted from the model to have a better result.
</p>

### 6.4. Model selection and parameter tuning

<p align="justify"> 
For model selection, I decided to choose a deep learning model tuned with Optuna library, for more information about optuna library you can check it out for more examples with keras in https://github.com/optuna/optuna-examples/tree/main/keras 

According to the notebook `Date_Fruit_Classification.ipynb` the steps to obtain the best model are the following:

  1. The function `MakeTrial` creates a trial with optuna library and based on the parameter ranges of my model  optuna evaluates the best accuracy result of my model according to these parameters.
  2. The function `Study_Statistics` shows the parameters of the best model such as number of hidden layers, activation function, learning rate, and so on.
  3. The function `MakeNeuralNetwork` creates a bigger model in epochs of the best model obtained, this is to see if the best model went into overfitting.
  4. The function `N_Models` puts all the previous steps together and creates a number of best models, this was done since optuna trial starts randomly and I wanted to have several models to analyze instead of one.
  5. The final step is the stability test, in that part I tested the stability of four models, giving them as input different test sets of different sizes.
  
The results show that in front of 150 different test sets the best model is the third one with a best accuracy value of `0.9333`, and the architecture of this model is:
- Number of hidden layers :2 
- Layer 1 number of neurons: 352
- Layer 1 activation function: elu
- Layer 1 dropout: 0.0
- Layer 2 number of neurons: 96
- Layer 2 activation function: selu
- Layer 2 dropout: 0.02
- Learning rate: 0.001760
- beta_1: 0.075162
- beta_2: 0.093541
- epsilon: 7.775386e-07

</p>
Model 3 Architecture: 
<p align="center">
  <img src="https://github.com/JesusAcuna/Date_Fruit_Classification_Keras/blob/main/images/model_3_architecture.png">
</p>

## 7. Instructions on how to run the project

Steps:
  1. Create an environment, this is explained here 4. [Setting up the virtual environment](#4-setting-up-the-virtual-environment)
  2. Run the file `train.py`, this  python file is modified, so that it doesn't take you a long time to train the models, with these parameters:
  
    - Number of models set to 2, line 390
    - Number of trials set to 1, line 391
    - Number of epochs set to 40, line 164
    - Number of epochs multiplier set to 2, linde 267
  
   The file `Date_Fruit_Classification.ipynb` was trained for 2 hours, for that I used a virtual machine on https://saturncloud.io/ with these parameters:
    
    - Number of models set to 4
    - Timeout of each model set to 1800 seconds
    - The range of epochs set to [300,350,400,450,500]
    - Number of epochs multiplier set to 6
  
   The output of `Date_Fruit_Classification.ipynb` are  the file `std_scaler.bin` and the directories `4_Models_000123`, `150_Stability_045011`.
   
   Inside `4_Models_000123` is the Model_3 directory with the file `Best_Model_3.h5`, which contains all the parameters of the best model I trained. I will put    it inside the repository to be able to do the next step.
   
  3. Run the file `converter_to_tflite.py` to convert the model `Best_Model_3.h5` to `Best_Model_3.tflite`, since the tensorFlow library is big and we need to use a tensorFlow lite library, which is a lighter library to predict.
    
  4. Run the file `predict.py` to run the web server locally. 
  
  5. Run the file `predict_test.py` to make a request to the web service, this file has an example labeled with class 'DOKOL'
  
  This is the result of the request: 
  
    {'BERHI': 5.466628351197495e-16, 'DEGLET': 3.063003077841131e-06, 'DOKOL': 0.9999969005584717, 'IRAQI': 3.4314470696553474e-25, 'ROTANA':    1.4219495495647376e-22, 'SAFAVI': 1.904230234707733e-23, 'SOGAY': 1.1417237294475413e-10}
  
  
## 8. Locally deployment 

Dockerfile contains all the specifications to build a container: python version, virtual environment, dependencies, scripts ,files, and so on. To do the cloud deployment we first need to configure locally and conteinerize it with Docker

Steps:

  1. Install it https://www.docker.com/, and if you're using WSL2 for running Linux as subsytem on Windows activate WSL Integration in Settings/Resources/WSL Integration.
  2. Open the console and locate in the repository where is the `Dockerfile` , if your using Windows there won't be any problem, but if you're using Linux
change two things in `Dockerfile`, first after this line `RUN pipenv install --system --deploy` type:

   RUN pipenv install gunicorn  
  
  and the second change the entrypoint for this:
    
   ENTRYPOINT ["gunicorn","--bind=0.0.0.0:9696","predict:app"]
  
  3. Build the docker and enter this command:

    docker build -t date_fruit_classification .
  
  4. Once you build the container you can chek all the images you created running this command:  `docker images`
  5. Run the docker entering this command:
  
  Windows
  
    winpty docker run -it --rm -p 9696:9696 date_fruit_classification:latest

  Linux
  
    docker run -it --rm -p 9696:9696 date_fruit_classification:latest
      
  <p align="center">
    <img src="https://github.com/JesusAcuna/Date_Fruit_Classification_Keras/blob/main/images/docker_run.png">
  </p>

  6. Activate the environment created with `pipenv shell`, and make a predict with `predict_test.py`, the result is as follows:
  
   <p align="center">
    <img src="https://github.com/JesusAcuna/Date_Fruit_Classification_Keras/blob/main/images/predict.png">
   </p>
  
  
## 9. Google Cloud deployment (GCP)

Steps:

  1. Create a Google Cloud Platform (GCP) account
  
  2. Install the gcloud CLI, you can follow the instrucctions here https://cloud.google.com/sdk/docs/install ,this is to be able to use gcloud console commands 
  
  3. Create a project:
    
    gcloud projects create date-fruit-classification 	

  4. To see all the projects you've created run the following:
  
    gcloud projects list 
    
  5. To select a project:
  
    gcloud config set project date-fruit-classification
    
    # To see what is the active project 
    gcloud config list project
    
  6. Create a tag to the image
  
    docker tag date_fruit_classification:latest gcr.io/date-fruit-classification/date-fruit-image:latest
    
  7. Activate Google Container Registry API 

    gcloud services enable containerregistry.googleapis.com
    
  8. To configure docker authentication run, this is for the next step : 
  
    gcloud auth configure-docker

  9. Push the  image to Google Container Registry 
  
    docker push gcr.io/date-fruit-classification/date-fruit-image:latest
    
  <p align="center">
    <img src="https://github.com/JesusAcuna/Date_Fruit_Classification_Keras/blob/main/images/container_registry.png">
   </p>    
   
  10. Deploy the image
   
    gcloud run deploy date-fruit-image --image gcr.io/date-fruit-classification/date-fruit-image:latest --port 9696 --max-instances 15 --platform managed --region us-central1 --allow-unauthenticated --memory 1Gi
    
  <p align="center">
    <img src="https://github.com/JesusAcuna/Date_Fruit_Classification_Keras/blob/main/images/google_cloud.png">
   </p>
    
  For more information on how to deploy : https://cloud.google.com/sdk/gcloud/reference/run/deploy
  
    #To delete a service
    gcloud run services delete date-fruit-image --region us-central1

  11. The web service was available on https://date-fruit-image-zpte776wvq-uc.a.run.app/predict, and the request I made with `predict_test_cloud.py` is in the image below, but if you do a deployment replace the URL they give you in `predict_test_cloud.py`.

  <p align="center">
    <img src="https://github.com/JesusAcuna/Date_Fruit_Classification_Keras/blob/main/images/predict_cloud.png">
  </p>
  
  
## 10. References

 Google Cloud Reference Documentation
 
 https://cloud.google.com/sdk/gcloud/reference
    
 Docker run reference
 
 https://docs.docker.com/engine/reference/run/
    
 Flask micro web framework
 
 https://flask.palletsprojects.com/en/2.2.x/
 
 Dataset Kaggle
 
 https://www.kaggle.com/datasets/muratkokludataset/date-fruit-datasets
 
 Optuna library
 
 https://optuna.readthedocs.io/en/stable/index.html
  
  
  
  

