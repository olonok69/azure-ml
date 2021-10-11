import mlflow
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.exceptions import ComputeTargetException
import os, shutil
from azureml.core import Experiment, ScriptRunConfig, Environment, Workspace, Dataset
from azureml.core.conda_dependencies import CondaDependencies
from azureml.widgets import RunDetails
import azureml.core
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import PipelineData, Pipeline
from azureml.pipeline.steps import PythonScriptStep
from azureml.core import Experiment
from azureml.core import Model
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig


# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))



# Create an Azure ML experiment in your workspace
experiment = Experiment(workspace = ws, name = "diabetes-training")
run = experiment.start_logging()
print("Starting experiment:", experiment.name)

# load the diabetes dataset
print("Loading Data...")
diabetes = pd.read_csv('data/diabetes.csv')

# Separate features and labels
X, y = diabetes[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, diabetes['Diabetic'].values

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Train a decision tree model
print('Training a decision tree model')
model = DecisionTreeClassifier().fit(X_train, y_train)

# calculate accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
print('Accuracy:', acc)
run.log('Accuracy', np.float(acc))

# calculate AUC
y_scores = model.predict_proba(X_test)
auc = roc_auc_score(y_test,y_scores[:,1])
print('AUC: ' + str(auc))
run.log('AUC', np.float(auc))

# Save the trained model
model_file = 'diabetes_model.pkl'
joblib.dump(value=model, filename=model_file)
run.upload_file(name = 'outputs/' + model_file, path_or_stream = './' + model_file)

# Complete the run
run.complete()

# Register the model
run.register_model(model_path='outputs/diabetes_model.pkl', model_name='diabetes_model',
                   tags={'Training context':'Inline Training'},
                   properties={'AUC': run.get_metrics()['AUC'], 'Accuracy': run.get_metrics()['Accuracy']})

print('Model trained and registered.')

# Deploy a Model as a Web Service
for model in Model.list(ws):
    print(model.name, 'version:', model.version)
    for tag_name in model.tags:
        tag = model.tags[tag_name]
        print ('\t',tag_name, ':', tag)
    for prop_name in model.properties:
        prop = model.properties[prop_name]
        print ('\t',prop_name, ':', prop)
    print('\n')

model = ws.models['diabetes_model']
print(model.name, 'version', model.version)


folder_name = 'diabetes_service'

# Create a folder for the web service files
experiment_folder = './' + folder_name
os.makedirs(experiment_folder, exist_ok=True)

print(folder_name, 'folder created.')

# Set path for scoring script
script_file = os.path.join(experiment_folder,"score_diabetes.py")


# Add the dependencies for our model (AzureML defaults is already included)
myenv = CondaDependencies()
myenv.add_conda_package('scikit-learn')

# Save the environment config as a .yml file
env_file = os.path.join(experiment_folder,"diabetes_env.yml")
with open(env_file,"w") as f:
    f.write(myenv.serialize_to_string())
print("Saved dependency info in", env_file)

# Print the .yml file
with open(env_file,"r") as f:
    print(f.read())

# Configure the scoring environment
inference_config = InferenceConfig(runtime= "python",
                                   entry_script=script_file,
                                   conda_file=env_file)

deployment_config = AciWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1)

service_name = "diabetes-service"

service = Model.deploy(ws, service_name, [model], inference_config, deployment_config)

service.wait_for_deployment(True)
print(service.state)


print(service.state)
print(service.get_logs())

for webservice_name in ws.webservices:
    print(webservice_name)


endpoint = service.scoring_uri
print(endpoint)


#### Use the Web Service


import json, requests

x_new = [[2,180,74,24,21,23.9091702,1.488172308,22]]
print ('Patient: {}'.format(x_new[0]))

# Convert the array to a serializable list in a JSON document
input_json = json.dumps({"data": x_new})

# Call the web service, passing the input data (the web service will also accept the data in binary format)
predictions = service.run(input_data = input_json)

# Get the predicted class - it'll be the first (and only) one.
predicted_classes = json.loads(predictions)
print(predicted_classes[0])


# This time our input is an array of two feature arrays
x_new = [[2,180,74,24,21,23.9091702,1.488172308,22],
         [0,148,58,11,179,39.19207553,0.160829008,45]]

# Convert the array or arrays to a serializable list in a JSON document
input_json = json.dumps({"data": x_new})

# Call the web service, passing the input data
predictions = service.run(input_data = input_json)

# Get the predicted classes.
predicted_classes = json.loads(predictions)
   
for i in range(len(x_new)):
    print ("Patient {}".format(x_new[i]), predicted_classes[i] )


x_new = [[2,180,74,24,21,23.9091702,1.488172308,22],
         [0,148,58,11,179,39.19207553,0.160829008,45]]

# Convert the array to a serializable list in a JSON document
input_json = json.dumps({"data": x_new})

# Set the content type
headers = { 'Content-Type':'application/json' }

predictions = requests.post(endpoint, input_json, headers = headers)
predicted_classes = json.loads(predictions.json())

for i in range(len(x_new)):
    print ("Patient {}".format(x_new[i]), predicted_classes[i] )