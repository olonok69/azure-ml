import pandas as pd
import mlflow
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.exceptions import ComputeTargetException
import os, shutil
from azureml.core import Experiment, ScriptRunConfig, Environment, Workspace
from azureml.core.conda_dependencies import CondaDependencies
from azureml.widgets import RunDetails
import azureml.core

# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))

# Create a folder for the experiment files
experiment_folder = 'diabetes_training_tree'
os.makedirs(experiment_folder, exist_ok=True)
print(experiment_folder, 'folder created')


compute_name = "gpu-cluster"

# checks to see if compute target already exists in workspace, else create it
try:
    compute_target = ComputeTarget(workspace=ws, name=compute_name)
except ComputeTargetException:
    config = AmlCompute.provisioning_configuration(vm_size="STANDARD_NC6",
                                                   vm_priority="lowpriority", 
                                                   min_nodes=0, 
                                                   max_nodes=1)

    compute_target = ComputeTarget.create(workspace=ws, name=compute_name, provisioning_configuration=config)
    compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)



# get the registered environment
registered_env = Environment.get(ws, 'diabetes-experiment-env')

# Get the training dataset
diabetes_ds = ws.datasets.get("diabetes dataset")

# Create a script config
script_config = ScriptRunConfig(source_directory=experiment_folder,
                              script='diabetes_training.py',
                              arguments = ['--input-data', diabetes_ds.as_named_input('training_data')], # Reference to dataset
                              environment=registered_env,
                              compute_target=compute_target) 

# submit the experiment
experiment_name = 'mslearn-train-diabetes'
experiment = Experiment(workspace=ws, name=experiment_name)
run = experiment.submit(config=script_config)

run.wait_for_completion()


# Get logged metrics
metrics = run.get_metrics()
for key in metrics.keys():
        print(key, metrics.get(key))
print('\n')
for file in run.get_file_names():
    print(file)

envs = Environment.list(workspace=ws)
for env in envs:
    print("Name",env)


for env in envs:
    if env.startswith("azureml"):
        print("Name",env)
        print("packages", envs[env].python.conda_dependencies.serialize_to_string())