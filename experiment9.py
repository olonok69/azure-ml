# https://github.com/MicrosoftLearning/DP100/blob/master/09A%20-%20Reviewing%20Automated%20Machine%20Learning%20Explanations.ipynb
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
try:
    # Prepare data for training
    default_ds = ws.get_default_datastore()
    if 'diabetes dataset' not in ws.datasets:
        default_ds.upload_files(files=['./data/diabetes.csv', './data/diabetes2.csv'], # Upload the diabetes csv files in /data
                            target_path='diabetes-data/', # Put it in a folder path in the datastore
                            overwrite=True, # Replace existing files of the same name
                            show_progress=True)

        # Create a tabular dataset from the path on the datastore (this may take a short while)
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
    train_data = ws.datasets.get("diabetes dataset")

    """
    Run an Automated Machine Learning Experiment
    To reduce time in this lab, you'll run an automated machine learning experiment with only three iterations.

    Note that the model_explainability configuration option is set to True.
    """
    # Configure Auto ML
    automl_config = AutoMLConfig(name='Automated ML Experiment',
                                task='classification',
                                compute_target=training_cluster,
                                training_data = train_data,
                                n_cross_validations = 2,
                                label_column_name='Diabetic',
                                iterations=3,
                                primary_metric = 'AUC_weighted',
                                max_concurrent_iterations=3,
                                featurization='off',
                                model_explainability=True # Generate feature importance!
                                )

    # Run the Auto ML experiment
    print('Submitting Auto ML experiment...')
    automl_experiment = Experiment(ws, 'diabetes_automl')
    automl_run = automl_experiment.submit(automl_config)
    automl_run.wait_for_completion(show_output=True)
    

except Exception as ex:
    print(ex)