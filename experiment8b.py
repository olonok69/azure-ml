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
from azureml.train.hyperdrive import GridParameterSampling, HyperDriveConfig, PrimaryMetricGoal, choice
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig
from azureml.train.automl import AutoMLConfig

# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))

# Prepare Data for an Experiment
default_ds = ws.get_default_datastore()

if 'diabetes dataset' not in ws.datasets:
    default_ds.upload_files(files=['./data/diabetes.csv', './data/diabetes2.csv'], # Upload the diabetes csv files in /data
                        target_path='diabetes-data/', # Put it in a folder path in the datastore
                        overwrite=True, # Replace existing files of the same name
                        show_progress=True)

    #Create a tabular dataset from the path on the datastore (this may take a short while)
    tab_data_set = Dataset.Tabular.from_delimited_files(path=(default_ds, 'diabetes-data/*.csv'))

    # Register the tabular dataset
    try:
        tab_data_set = tab_data_set.register(workspace=ws, 
                                name='diabetes dataset',
                                description='diabetes data',
                                tags = {'format':'CSV'},
                                create_new_version=True)
        print('Dataset registered.')
    except Exception as ex:
        print(ex)
else:
    print('Dataset already registered.')


# Split the dataset into training and validation subsets
diabetes_ds = ws.datasets.get("diabetes dataset")
train_ds, test_ds = diabetes_ds.random_split(percentage=0.7, seed=123)
print("Data ready!")


cluster_name = "cpu-cluster"

try:
    # Check for existing compute target
    training_cluster = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    # If it doesn't already exist, create it
    try:
        compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS11_V2', max_nodes=2)
        training_cluster = ComputeTarget.create(ws, cluster_name, compute_config)
        training_cluster.wait_for_completion(show_output=True)
    except Exception as ex:
        print(ex)

# Configure Automated Machine Learning
automl_config = AutoMLConfig(name='Automated ML Experiment',
                             task='classification',
                             compute_target=training_cluster,
                             training_data = train_ds,
                             validation_data = test_ds,
                             label_column_name='Diabetic',
                             iterations=6,
                             primary_metric = 'AUC_weighted',
                             max_concurrent_iterations=4,
                             featurization='auto'
                             )

print("Ready for Auto ML run.")

# Run an Automated Machine Learning Experiment
print('Submitting Auto ML experiment...')
automl_experiment = Experiment(ws, 'diabetes_automl')
automl_run = automl_experiment.submit(automl_config)

automl_run.wait_for_completion(show_output=True)

# Determine the Best Performing Model
best_run, fitted_model = automl_run.get_output()
print(best_run)
print(fitted_model)
best_run_metrics = best_run.get_metrics()
for metric_name in best_run_metrics:
    metric = best_run_metrics[metric_name]
    print(metric_name, metric)

for step in fitted_model.named_steps:
    print(step)


# Register model
best_run.register_model(model_path='outputs/model.pkl', model_name='diabetes_model_automl',
                        tags={'Training context':'Auto ML'},
                        properties={'AUC': best_run_metrics['AUC_weighted'], 'Accuracy': best_run_metrics['accuracy']})

# List registered models
for model in Model.list(ws):
    print(model.name, 'version:', model.version)
    for tag_name in model.tags:
        tag = model.tags[tag_name]
        print ('\t',tag_name, ':', tag)
    for prop_name in model.properties:
        prop = model.properties[prop_name]
        print ('\t',prop_name, ':', prop)
    print('\n')