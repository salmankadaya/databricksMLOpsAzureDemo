# Databricks notebook source
# MAGIC %md
# MAGIC Importing the Dependencies

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# COMMAND ----------

# MAGIC %md
# MAGIC Data Collection and Analysis

# COMMAND ----------

# loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv('/dbfs/FileStore/shared_uploads/salmankadaya@gmail.com/diabetes.csv') 

# COMMAND ----------

# printing the first 5 rows of the dataset
diabetes_dataset.head()

# COMMAND ----------

# separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']
print(X)
print(Y)

# COMMAND ----------

# MAGIC %md
# MAGIC Train Test Split

# COMMAND ----------

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC Training the Model

# COMMAND ----------

classifier = svm.SVC(kernel='linear')
#training the support vector Machine Classifier
model = classifier.fit(X_train, Y_train)

# COMMAND ----------

y_pred = model.predict(X_test)
print(y_pred)
#y_pred_prob = model.predict_proba(X_test)
#print(y_pred_prob)

# COMMAND ----------

# MAGIC %md
# MAGIC Model Evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC Accuracy Score

# COMMAND ----------

# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)

# COMMAND ----------

# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC Making a Predictive System

# COMMAND ----------

input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = classifier.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')

# COMMAND ----------

# MAGIC %md
# MAGIC Saving the trained model

# COMMAND ----------

import pickle
filename = 'trained_model.pkl'
pickle.dump(classifier, open(filename, 'wb'))

# COMMAND ----------

# loading the saved model
loaded_model = pickle.load(open('trained_model.pkl', 'rb'))

# COMMAND ----------

input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')

# COMMAND ----------

!mlflow --version

# COMMAND ----------

def get_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score,precision_score,recall_score,log_loss
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred,average='micro')
    recall = recall_score(y_true, y_pred,average='micro')
    return {'accuracy': round(acc, 2), 'precision': round(prec, 2), 'recall': round(recall, 2)}

# COMMAND ----------

run_metrics = get_metrics(Y_test, y_pred)
run_metrics

# COMMAND ----------

def create_confusion_matrix_plot(clf, X_test, Y_test):
    import matplotlib.pyplot as plt
    from sklearn.metrics import plot_confusion_matrix
    plot_confusion_matrix(clf, X_test, Y_test)
    plt.savefig('confusion_matrix.png')

# COMMAND ----------

create_confusion_matrix_plot(model, X_test, Y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC Define create_experiment function to track your model experiment within MLFlow
# MAGIC Adding an MLflow Model to the Model Registry
# MAGIC Execute the create_experiment function and log experiment

# COMMAND ----------

def create_experiment(experiment_name,run_name, run_metrics,model, confusion_matrix_path = None, 
                      roc_auc_plot_path = None, run_params=None):
    import mlflow
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name):
        
        if not run_params == None:
            for param in run_params:
                mlflow.log_param(param, run_params[param])
            
        for metric in run_metrics:
            mlflow.log_metric(metric, run_metrics[metric])
        
        
        
        if not confusion_matrix_path == None:
            mlflow.log_artifact(confusion_matrix_path, 'confusion_materix')
            
        if not roc_auc_plot_path == None:
            mlflow.log_artifact(roc_auc_plot_path, "roc_auc_plot")
        
        mlflow.set_tag("tag1", "Iris Classifier")
        mlflow.set_tags({"tag2":"Logistic Regression", "tag3":"Multiclassification using Ovr - One vs rest class"})
        mlflow.sklearn.log_model(model, "model" ,registered_model_name="iris-classifier")
    print('Run - %s is logged to Experiment - %s' %(run_name, experiment_name))

# COMMAND ----------

from datetime import datetime
experiment_name = "/Users/salmankadaya@gmail.com/MLOps/iris_classifier_"+ str(datetime.now().strftime("%d-%m-%y")) ##basic classifier
run_name="iris_classifier_"+str(datetime.now().strftime("%d-%m-%y"))
create_experiment(experiment_name,run_name,run_metrics,model,'confusion_matrix.png')

# COMMAND ----------

import mlflow
logged_model = 'runs:/b52ef8dac45243458ec1b9eacbe3204e/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
loaded_model.predict(pd.DataFrame(X_test))




