# Importing necessary modules
import requests, zipfile
from io import BytesIO
#
import tensorflow as tf
from tensorflow import keras
#
import matplotlib.pyplot as plt

# Defining the zip file URL
url = 'https://www.muratkoklu.com/datasets/vtdhnd06.php'

# Downloading the file by sending the request to the URL
req = requests.get(url)

# Extracting the zip file contents
zipfile= zipfile.ZipFile(BytesIO(req.content))
zipfile.extractall('./') #Current directory

# Data set name
dataset_complete_name=zipfile.namelist()[2]
dataset_complete_name
start = dataset_complete_name.index('/')
end = dataset_complete_name.index('.',start+1)

dataset_name = dataset_complete_name[start+1:end]

"""### Getting Data from xlsx File"""
import pandas as pd               # Importing pandas library for dataframes
import numpy as np                # Importing numpy library for array operations

# Selecting the xlsx file path 
df = pd.read_excel(f"./{dataset_complete_name}",sheet_name=0) # sheet_name = 0, the first sheet was selected

#%% Changing the target variable from object to numeric

classes=list((df['Class'].value_counts().index))

n_classes=np.arange(len(classes))
i_classes=dict(zip(classes,n_classes))

df['Class']=df['Class'].map(i_classes)


#%% Splitting Data: Training and Test

from sklearn.model_selection import train_test_split
X=df.iloc[:,0:-1].values
y=df.iloc[:,-1].values

x_train, x_test,y_train,y_test= train_test_split(X,y,            
                                                 test_size=0.2,  # test_size=0.2  , 20% testing and 80% training 
                                                 random_state=1, # random_state=0 , shuffle the data, reproducible results
                                                 stratify=y)     # stratify=y     , makes the values of the target variable 
                                                                 #                  equivalent in quantity for training and testing

#%% Nomalization

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train= sc.fit_transform(x_train)
X_test= sc.transform(x_test)

# Save StandardScaler object
import joblib
joblib.dump(sc, 'std_scaler.bin', compress=True)


from keras.utils import to_categorical 
# Y_train
Y_train = to_categorical(y_train)

# Y_test
Y_test = to_categorical(y_test)

#%%
# Multi Layer Perceptron (MLP)
## Hyperparameter Tuning-Optuna

#!pip install --quiet optuna
import optuna

from optuna.integration import TFKerasPruningCallback
from optuna.trial import TrialState

# keras.layers
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout

# keras.models
from keras.models import Sequential

# keras.utils
from keras.utils import plot_model

# keras.optimizers
from keras.optimizers import adam_v2

""" Define objective function for the structure of the MLP """

def create_model(trial):

    n_layers = trial.suggest_int("n_layers", 1, 8,step=1)                 # Number of hidden layers
    model = Sequential()                                                  # Model creation : Instantiate the Sequential object
    model.add(Input(shape=(34,)))                                         # Input layer (Number of independent variables)
    # Number of neurons and activation function of each Hidden layer
    for i in range(n_layers):
        # Number of neurons for the ith hidden layer
        num_hidden = trial.suggest_int(f'n_units_L{i}',                   # Number of neurons for the ith hidden layer
                                       32, 512, step = 32)
        # Activation function for the ith hidden layer
        func_activation = trial.suggest_categorical( f'f_activation_L{i}',
                                                    ['relu','sigmoid','tanh','selu','elu'])
        model.add(Dense(units=num_hidden,
                        activation=func_activation))
        # Dropout Layer
        dropout = trial.suggest_categorical(f'dropout_L{i}',              
                                            [0.0,0.2,0.4,0.6,0.8])
        
        model.add(Dropout(rate=dropout))

    # Output layer (Number of dependent variables)
    model.add(Dense(units=7, 
                    activation="softmax"))

    # Compile the model with a sampled learning rate.
    learning_rate = trial.suggest_float("learning_rate",                  # Range of learning rate values
                                        1e-6, 1e-1,
                                        log=True) 
    beta_1 = trial.suggest_float("beta_1",                                # Range of beta_1 values
                                 1e-3, 1e-1, log=True)
    beta_2 = trial.suggest_float("beta_2",                                # Range of beta_2 values
                                 1e-8, 1e-1, log=True)
    epsilon = trial.suggest_float("epsilon",                              # Range of epsilon values
                                  1e-11, 1e-6, log=True)

    model.compile(loss="categorical_crossentropy",
                  optimizer=adam_v2.Adam(learning_rate = learning_rate,
                                         beta_1=beta_1,
                                         beta_2=beta_2,
                                         epsilon=epsilon),
                  metrics="accuracy")
    return model

"""  Define objective function for Optuna """

def objective(trial):

    # Generate our trial model
    model = create_model(trial)

    # Fit the model on the training data.
    # The TFKerasPruningCallback checks for pruning condition every epoch.
    """
    -PruningCallback stops unpromising trials at the early stages 
    of the training based on "val_accuracy" in this case.

    -The model is fitted based on the validation set.
    """
    model.fit(X_train,Y_train,
              batch_size=20,
              callbacks=[TFKerasPruningCallback(trial, "val_accuracy")],
              epochs= trial.suggest_categorical("epochs",[40] ), #[300,350,400,450,500]
              validation_data=(X_test, Y_test),
              verbose=1)

    # Evaluate the model accuracy on the validation set in each trial
    score = model.evaluate(X_test, Y_test, verbose=0)

    return score[1]

#%% MakeTrial

def MakeTrial(study_name,n_trials=None,timeout=None):
  
  print(f"\nStarting Trial: {study_name}\n")

  study = optuna.create_study(study_name=study_name,
                              direction="maximize",
                              pruner=optuna.pruners.MedianPruner())
  
  # Set 'n_trials' and/or 'timeout' in seconds for optimization 
  study.optimize(objective, n_trials=n_trials, timeout=timeout)
  pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
  complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

  print(f"\nTrial Completed: {study_name}\n")

  return study, pruned_trials, complete_trials

#%% Study_Statistics

def Study_Statistics(study, pruned_trials, complete_trials):
  print(f"{study.study_name} study statistics:")
  print("  Number of finished trials: ", len(study.trials))
  print("  Number of pruned trials: ", len(pruned_trials))
  print("  Number of complete trials: ", len(complete_trials))
  # Best trial
  print("Best trial:")
  trial = study.best_trial
  # Accuracy value
  print("  Value: ", trial.value)
  # Best trial parameters
  print("  Params: ")
  for key, value in trial.params.items():
      print(f"    {key}: {value}")


#%% MakeNeuralNetwork
# Creating a larger model in epochs 

# This callback allows to get the best model when a model is being fitted
from keras.callbacks import ModelCheckpoint

# This function is similar to 'create_model(trial)' function 
# Also this function accepts as arguments the previous study 
def MakeNeuralNetwork(study,
                      X_train,Y_train,
                      X_val,Y_val,
                      verbose,
                      path='./best_model.h5' ):
    model=Sequential()                                       # Model creation : Instantiate the Sequential object                 
    # Add layers to my Neural Network
    # Each hidden layer must have its activation function
    model.add(Input(shape=(34,),                             # Number of input features
                    name="InputLayer"))
    
    for i in range(study.best_trial.params['n_layers']):
      model.add(Dense(
                  # Number of neurons for the ith hidden layer
                  units= study.best_trial.params[f'n_units_L{i}'],
                  # Activation function
                  activation=study.best_trial.params[f'f_activation_L{i}'],
                  #  We can add an alias to the ith hidden layer
                  name= f'{i+1}HiddenLayer_'+str(study.best_trial.params[f'n_units_L{i}'])+'Neurons'))
      
      model.add(Dropout(rate= study.best_trial.params[f'dropout_L{i}'],       # Dropout layer rate
                        name=f"{i+1}Dropout"))
  
    # At the end I add one last layer: Output Layer
    # Remember that the dependent variable is multiclass
    
    model.add(Dense(units=7,
                     activation="softmax",
                     name="OutputLayer"))
    
    # Model compilation
    model.compile(loss="categorical_crossentropy",
                  optimizer=adam_v2.Adam(learning_rate = study.best_trial.params['learning_rate'],
                                         beta_1 = study.best_trial.params['beta_1'],
                                         beta_2 = study.best_trial.params['beta_2'],
                                         epsilon = study.best_trial.params['epsilon']),
                  metrics="accuracy")
    
    # Path where our 'model.h5' file is going to be saved 
    checkpoint_filepath = path

    model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath,
                                                monitor='val_accuracy',
                                                mode='max',
                                                save_best_only=True)
    
    history=model.fit(X_train,Y_train,
                      validation_data=(X_val, Y_val),
                      batch_size=20,
                      epochs= int(study.best_trial.params['epochs']*2),#6         # Multiplier of the number of epochs
                      verbose=verbose,
                      callbacks=[model_checkpoint_callback])

    return model,history


#%% Model History Plot

def plot_save_history(model_history):

  fig,ax=plt.subplots(2,1,figsize=(15,8))
  fig.subplots_adjust(hspace=0.4)

  ax[0].plot(model_history.history["val_loss"])
  ax[0].set_title("Loss_Function_val",fontsize=16)
  ax[0].set_xlabel("Epochs",fontsize=16)
  ax[0].set_ylabel("Loss",fontsize=16)
  
  ax[1].plot(model_history.history["val_accuracy"])
  ax[1].set_title("Accuracy_val",fontsize=16)
  ax[1].set_xlabel("Epochs",fontsize=16)
  ax[1].set_ylabel("Acc",fontsize=16)
  ax[1].set_ylim([0, 1])

  plt.close()
  
  return fig

#%%  Saving N_Models

import os
import datetime 

def N_Models(n,n_trials=None,timeout=None):
  """
    n: Number of models
    n_trials: Number of trials
    timeout: Time in seconds
  """

  # Study List: best parameters and value list 
  best_params_list=[]
  value_list=[]

  # Create a directory for n Models
  now = datetime.datetime.now()
  time_HMS = now.strftime("%H%M%S")
  DirModels = f"{n}_Models_{time_HMS}/" 
  os.makedirs(DirModels)
  
  for i in range(n):
    #
    study_name = f"Model_{i+1}"
    # Set 'n_trials' and/or 'timeout' in seconds for optimization 
    study, pruned_trials, complete_trials = MakeTrial(study_name, n_trials=n_trials,timeout=timeout)

    # Study List
    best_params_list.append(study.best_trial.params)
    value_list.append(study.best_trial.value)
    
    # Create a subdirectory for each model
    SubDirModel = DirModels + f"Model_{i+1}/" 
    os.makedirs(SubDirModel)

    # Creating models
    print(f"\nBuilding Model_{i+1}\n")
    model, model_history = MakeNeuralNetwork(study,
                                            X_train,Y_train,
                                            X_test,Y_test,
                                            verbose=1,
                                            path = SubDirModel + f"Best_Model_{i+1}.h5" ) 

    # Architecture Model Visualization
    plot_model(model,
               show_shapes=True,
               show_layer_names=True,
               show_layer_activations=True,
               to_file=SubDirModel + f"Model_{i+1}_Architecture.jpg")

    # Model History Plot
    fig=plot_save_history(model_history)

    # History Files (jpg and csv)
    history_name = f"Model_{i+1}_History"
    history_name_jpg = SubDirModel + history_name + ".jpg"
    history_name_csv = SubDirModel + history_name + ".csv"

    # History DataFrame
    df_history = pd.DataFrame(model_history.history)

    # Saving History Files (csv and jpg)
    df_history.to_csv(history_name_csv)
    fig.savefig(history_name_jpg)


  # Making a dataframe of all models
  max_n_elements,idx=max([(len(i),index) for index,i in enumerate(best_params_list)])

  columns=list(best_params_list[idx].keys())

  list_models_left=[list(best_params_list[i].values())[:-2] for i in range(n)]
  list_models_right=[list(best_params_list[i].values())[-2:] for i in range(n)]

  df_models_left=pd.DataFrame(list_models_left)
  df_models_right=pd.DataFrame(list_models_right)

  df_models=pd.concat([df_models_left, df_models_right], axis=1)

  df_models.columns=columns
  df_models.fillna(" ",inplace=True)
  df_models['acc']=value_list                                # accuracy value from Optuna Trial
  df_models['model']=[f"Model_{i+1}" for i in range(n)]

  df_models.set_index('model',inplace=True)

  df_models.to_csv(DirModels + f"{i+1}_Models.csv")

  return df_models, DirModels

#%% I chose 4 models with a 30-minute fitting

# Set 'n_trials' and/or 'timeout' in seconds for optimization 
n=2
df_models, DirModels = N_Models(n=n,n_trials=1) #n=4,tiemout=1800

print(df_models)
#%% Stability Test of our MLP Model
#Function that allows me to generate random samples

def MakeSample(X,y, SampleSize):

  Rows = np.random.randint(0, pd.DataFrame(X).shape[0], size = SampleSize)
  # 
  X = pd.DataFrame(X).iloc[Rows, :]
  y = pd.DataFrame(y).iloc[Rows, :]

  # Output 
  return X,y

#%%

#Let's analyze the stability of our neural network 
import datetime
import os
from keras.models import load_model

# Number of experiments
N_Experiments = 150
SampleSizeList = []
LossValueList = []
AccValueList = []

# time_HMS generate a string with the hour,minute and seconds
now = datetime.datetime.now()
time_HMS = now.strftime("%H%M%S")

# Stability test directory
DirExp = f"{N_Experiments}_Stability_{time_HMS}/" 
os.makedirs(DirExp)

for i in range(n):
  model = load_model(DirModels + f"/Model_{i+1}/Best_Model_{i+1}.h5")

  SubDirExp= DirExp + f"Model_{i+1}/" 
  os.makedirs(SubDirExp)

  SampleSizeL= []
  LossValueL = []
  AccValueL = []

  for j in range(N_Experiments):
    """
    - Generate a random sammple size
    - Choose min and max values based on the number of validation examples
    - The size of our test data is 180
    """
    SampleSize = np.random.randint(30, 180, 1)[0]

    # MakeSample function
    xm,ym = MakeSample(X_test,Y_test, SampleSize)

    # Save these examples
    xmName = SubDirExp + f"DepVarSample_{SampleSize}.csv"
    xm.to_csv(xmName)

    ymName = SubDirExp + f"IndepVarSample_{SampleSize}.csv"
    ym.to_csv(ymName)

    # Evaluate model loss value and accuracy 
    output = model.evaluate(xm,ym)

    # Save all the results in the appropriate lists
    SampleSizeL.append(SampleSize)
    LossValueL.append(output[0])
    AccValueL.append(output[1])

  SampleSizeList.append(SampleSizeL)
  LossValueList.append(LossValueL)
  AccValueList.append(AccValueL)

  # Pack the results into a dataframe
  Model_Stability = pd.DataFrame({"SampleSize" : SampleSizeL,
                                    "LossFunction" : LossValueL,
                                    "Accuracy" : AccValueL} )

  Model_Stability.to_csv(DirExp + f"/Model_Stability_{i+1}.csv",index=False)

#%% Box plot

import matplotlib.pyplot as plt  
import seaborn as sns 

Models_Accuracy=[pd.read_csv(DirExp + f"Model_Stability_{i+1}.csv")['Accuracy'].values for i in range(n)]

Models_Keys=[f"Model_{i+1}" for i in range(n)]

data = pd.DataFrame(dict(zip(Models_Keys,Models_Accuracy)))

fig, ax = plt.subplots(figsize=(15, 10))
sns.set_style('darkgrid')

sns.boxplot(data = data,width=0.3)
ax.set_title("Model")

plt.show()
