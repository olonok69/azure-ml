# Check core SDK version number
import azureml.core
from azureml.telemetry import set_diagnostics_collection
from azureml.core.workspace import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil
from azureml.core import Experiment, Environment, ScriptRunConfig
from azureml.core.dataset import Dataset
import urllib.request
from azureml.exceptions import UserErrorException
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
import pkg_resources
from azureml.interpret import ExplanationClient
from azureml.core.model import Model
import joblib
from raiwidgets import ExplanationDashboard

# make sure utils.py is in the same directory as this code


print("SDK version:", azureml.core.VERSION)

set_diagnostics_collection(send_diagnostics=True)

ws = Workspace.from_config()
print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep = '\n')


# Choose a name for your CPU cluster
cpu_cluster_name = "cpu-cluster"

# Verify that cluster does not exist already
try:
    cpu_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2',
                                                           max_nodes=4)
    cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config)

cpu_cluster.wait_for_completion(show_output=True)

experiment_name = 'explainer-remote-run-on-amlcompute'
experiment = Experiment(workspace=ws, name=experiment_name)


project_folder = './explainer-remote-run-on-amlcompute'
os.makedirs(project_folder, exist_ok=True)
shutil.copy('train_explain.py', project_folder)


# Create a new RunConfig object
run_config = RunConfiguration(framework="python")

# Set compute target to AmlCompute target created in previous step
run_config.target = cpu_cluster.name

azureml_pip_packages = [
    'azureml-defaults', 'azureml-telemetry', 'azureml-interpret'
]

# Note: this is to pin the scikit-learn and pandas versions to be same as notebook.
# In production scenario user would choose their dependencies

available_packages = pkg_resources.working_set
sklearn_ver = None
pandas_ver = None
for dist in list(available_packages):
    if dist.key == 'scikit-learn':
        sklearn_ver = dist.version
    elif dist.key == 'pandas':
        pandas_ver = dist.version
sklearn_dep = 'scikit-learn'
pandas_dep = 'pandas'
if sklearn_ver:
    sklearn_dep = 'scikit-learn=={}'.format(sklearn_ver)
if pandas_ver:
    pandas_dep = 'pandas=={}'.format(pandas_ver)
# Specify CondaDependencies obj
# The CondaDependencies specifies the conda and pip packages that are installed in the environment
# the submitted job is run in.  Note the remote environment(s) needs to be similar to the local
# environment, otherwise if a model is trained or deployed in a different environment this can
# cause errors.  Please take extra care when specifying your dependencies in a production environment.
azureml_pip_packages.extend([sklearn_dep, pandas_dep])
run_config.environment.python.conda_dependencies = CondaDependencies.create(pip_packages=azureml_pip_packages)


src = ScriptRunConfig(source_directory=project_folder, 
                      script='train_explain.py', 
                      run_config=run_config) 
run = experiment.submit(config=src)

# Shows output of the run on stdout.
run.wait_for_completion(show_output=True)

print(run.get_metrics())

# Download

# Get model explanation data
client = ExplanationClient.from_run(run)
global_explanation = client.download_model_explanation()
local_importance_values = global_explanation.local_importance_values
expected_values = global_explanation.expected_values

# Or you can use the saved run.id to retrive the feature importance values
client = ExplanationClient.from_run_id(ws, experiment_name, run.id)
global_explanation = client.download_model_explanation()
local_importance_values = global_explanation.local_importance_values
expected_values = global_explanation.expected_values

# Get the top k (e.g., 4) most important features with their importance values
global_explanation_topk = client.download_model_explanation(top_k=4)
global_importance_values = global_explanation_topk.get_ranked_global_values()
global_importance_names = global_explanation_topk.get_ranked_global_names()

print('global importance values: {}'.format(global_importance_values))
print('global importance names: {}'.format(global_importance_names))

# Retrieve model for visualization and deployment

original_model = Model(ws, 'model_explain_model_on_amlcomp')
model_path = original_model.download(exist_ok=True)
original_model = joblib.load(model_path)

# Retrieve x_test for visualization
x_test_path = './x_test_boston_housing.pkl'
run.download_file('x_test_boston_housing.pkl', output_file_path=x_test_path)

x_test = joblib.load('x_test_boston_housing.pkl')

ExplanationDashboard(global_explanation, original_model, dataset=x_test)