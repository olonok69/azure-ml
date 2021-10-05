#https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/ml-frameworks/scikit-learn/train-hyperparameter-tune-deploy-with-sklearn/train-hyperparameter-tune-deploy-with-sklearn.ipynb


# Check core SDK version number
import azureml.core
from azureml.telemetry import set_diagnostics_collection
from azureml.core.workspace import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
import os
import shutil
from azureml.core import Experiment, Environment, ScriptRunConfig
from azureml.train.hyperdrive.runconfig import HyperDriveConfig
from azureml.train.hyperdrive.sampling import RandomParameterSampling
from azureml.train.hyperdrive.run import PrimaryMetricGoal
from azureml.train.hyperdrive.parameter_expressions import choice

print("SDK version:", azureml.core.VERSION)

set_diagnostics_collection(send_diagnostics=True)

ws = Workspace.from_config()
print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep = '\n')

# choose a name for your cluster
cluster_name = "hd-cluster"

try:
    compute_target = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing compute target')
except ComputeTargetException:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2', 
                                                           max_nodes=4)

    # create the cluster
    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)

# can poll for a minimum number of nodes and for a specific timeout. 
# if no min node count is provided it uses the scale settings for the cluster
compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

# use get_status() to get a detailed status for the current cluster. 
print(compute_target.get_status().serialize())


project_folder = './sklearn-iris'
os.makedirs(project_folder, exist_ok=True)
shutil.copy('train_iris.py', project_folder)

# Create an experiment
experiment_name = 'train_iris'
experiment = Experiment(ws, name=experiment_name)

# Define a conda environment YAML file with your training script dependencies and create an Azure ML environment.
sklearn_env = Environment.from_conda_specification(name = 'sklearn-env', file_path = './conda_dependencies.yml')

#Create a ScriptRunConfig object to specify the configuration details of your training job, including your training script, environment to use, and the compute target to run on.

src = ScriptRunConfig(source_directory=project_folder,
                      script='train_iris.py',
                      arguments=['--kernel', 'linear', '--penalty', 1.0],
                      compute_target=compute_target,
                      environment=sklearn_env)

# uncomment to run a single experiment
# run = experiment.submit(src)
# run.wait_for_completion(show_output=True)

param_sampling = RandomParameterSampling( {
    "--kernel": choice('linear', 'rbf', 'poly', 'sigmoid'),
    "--penalty": choice(0.5, 1, 1.5)
    }
)

hyperdrive_config = HyperDriveConfig(run_config=src,
                                     hyperparameter_sampling=param_sampling, 
                                     primary_metric_name='Accuracy',
                                     primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                                     max_total_runs=12,
                                     max_concurrent_runs=4)


# start the HyperDrive run
hyperdrive_run = experiment.submit(hyperdrive_config)

hyperdrive_run.wait_for_completion(show_output=True)

assert(hyperdrive_run.get_status() == "Completed")

best_run = hyperdrive_run.get_best_run_by_primary_metric()
print(best_run.get_details()['runDefinition']['arguments'])

print(best_run.get_file_names())

model = best_run.register_model(model_name='sklearn-iris', model_path='outputs/model.joblib')