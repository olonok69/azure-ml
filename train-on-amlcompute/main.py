# Check core SDK version number
# https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/training/train-on-amlcompute/train-on-amlcompute.ipynb
import azureml.core
from azureml.core import Workspace, Experiment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core import Environment
from azureml.core.runconfig import DockerConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import ScriptRunConfig
import os

print("SDK version:", azureml.core.VERSION)

ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep = '\n')

experiment_name = 'train-on-amlcompute'
experiment = Experiment(workspace = ws, name = experiment_name)

AmlCompute.supported_vmsizes(workspace = ws)
#AmlCompute.supported_vmsizes(workspace = ws, location='southcentralus')

myenv = Environment("myenv")
myenv.python.conda_dependencies = CondaDependencies.create(conda_packages=['scikit-learn', 'packaging'])

# Enable Docker
docker_config = DockerConfiguration(use_docker=True)

# Choose a name for your CPU cluster
cpu_cluster_name = "cpu-cluster"


project_folder = './'


# Verify that cluster does not exist already
try:
    cpu_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2',
                                                           max_nodes=4)
    cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config)

cpu_cluster.wait_for_completion(show_output=True)

src = ScriptRunConfig(source_directory=project_folder, 
                      script='train.py', 
                      compute_target=cpu_cluster, 
                      environment=myenv,
                      docker_runtime_config=docker_config)
 
run = experiment.submit(config=src)

run.wait_for_completion(show_output=True)

run.get_metrics()
