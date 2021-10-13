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


# Prepare a Training Script

experiment_folder = 'diabetes_training-hyperdrive'
os.makedirs(experiment_folder, exist_ok=True)

print('Folder ready.')


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

# Run a Hyperdrive Experiment
# Create a Python environment for the experiment
sklearn_env = Environment("sklearn-env")

# Ensure the required packages are installed (we need scikit-learn, Azure ML defaults, and Azure ML dataprep)
packages = CondaDependencies.create(pip_packages=['scikit-learn','azureml-defaults','azureml-dataprep[pandas]'])
sklearn_env.python.conda_dependencies = packages

# Get the training dataset
diabetes_ds = ws.datasets.get("diabetes dataset")

# Create a script config
script_config = ScriptRunConfig(source_directory=experiment_folder,
                              script='diabetes_training.py',
                              arguments = ['--regularization', 0.1, # Regularizaton rate parameter
                                           '--input-data', diabetes_ds.as_named_input('training_data')], # Reference to dataset
                              environment=sklearn_env,
                              compute_target = training_cluster)

# Sample a range of parameter values
params = GridParameterSampling(
    {
        # There's only one parameter, so grid sampling will try each value - with multiple parameters it would try every combination
        '--regularization': choice(0.001, 0.005, 0.01, 0.05, 0.1, 1.0)
    }
)

# Configure hyperdrive settings
hyperdrive = HyperDriveConfig(run_config=script_config, 
                          hyperparameter_sampling=params, 
                          policy=None, 
                          primary_metric_name='AUC', 
                          primary_metric_goal=PrimaryMetricGoal.MAXIMIZE, 
                          max_total_runs=6,
                          max_concurrent_runs=4)

# Run the experiment
experiment = Experiment(workspace = ws, name = 'diabates_training_hyperdrive')
run = experiment.submit(config=hyperdrive)

# Show the status in the notebook as the experiment runs

run.wait_for_completion()

# Determine the Best Performing Run

for child_run in run.get_children_sorted_by_primary_metric():
    print(child_run)

best_run = run.get_best_run_by_primary_metric()
best_run_metrics = best_run.get_metrics()
parameter_values = best_run.get_details() ['runDefinition']['arguments']

print('Best Run Id: ', best_run.id)
print(' -AUC:', best_run_metrics['AUC'])
print(' -Accuracy:', best_run_metrics['Accuracy'])
print(' -Regularization Rate:',parameter_values)


# Register model
best_run.register_model(model_path='outputs/diabetes_model.pkl', model_name='diabetes_model',
                        tags={'Training context':'Hyperdrive'},
                        properties={'AUC': best_run_metrics['AUC'], 'Accuracy': best_run_metrics['Accuracy']})

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