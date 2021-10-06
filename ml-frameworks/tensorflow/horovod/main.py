# Check core SDK version number
import azureml.core
from azureml.telemetry import set_diagnostics_collection
from azureml.core.workspace import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import Dataset
import os, shutil
from azureml.core import Experiment, Environment
from azureml.core import ScriptRunConfig
from azureml.core.runconfig import MpiConfiguration


print("SDK version:", azureml.core.VERSION)

set_diagnostics_collection(send_diagnostics=True)

ws = Workspace.from_config()
print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep='\n')

# choose a name for your cluster
cluster_name = "gpu-cluster"

try:
    compute_target = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing compute target')
except ComputeTargetException:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_NC6', 
                                                           max_nodes=4)

    # create the cluster
    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)

    compute_target.wait_for_completion(show_output=True)

# use get_status() to get a detailed status for the current cluster. 
print(compute_target.get_status().serialize())

# Create a Dataset for Files
web_paths = ['https://azureopendatastorage.blob.core.windows.net/testpublic/text8.zip']
dataset = Dataset.File.from_files(path=web_paths)

dataset = dataset.register(workspace=ws,
                           name='wikipedia-text',
                           description='Wikipedia text training and test dataset',
                           create_new_version=True)

# list the files referenced by the dataset
print(dataset.to_path())

# Train model on the remote compute

project_folder = './tf-distr-hvd'
os.makedirs(project_folder, exist_ok=True)
shutil.copy('tf_horovod_word2vec.py', project_folder)
# Create an experiment
experiment_name = 'tf-distr-hvd'
experiment = Experiment(ws, name=experiment_name)
# Create an environment
tf_env = Environment.get(ws, name='AzureML-TensorFlow-1.13-GPU')
"""
Configure the training job
Create a ScriptRunConfig object to specify the configuration details of your training job, including your training script, environment to use, 
and the compute target to run on.
In order to execute a distributed run using MPI/Horovod, you must create an MpiConfiguration object and pass it to the distributed_job_config 
parameter of the ScriptRunConfig constructor. The below code will configure a 2-node distributed job running one process per node. 
If you would also like to run multiple processes per node (i.e. if your cluster SKU has multiple GPUs), additionally specify the 
process_count_per_node parameter in MpiConfiguration (the default is 1).
"""

src = ScriptRunConfig(source_directory=project_folder,
                      script='tf_horovod_word2vec.py',
                      arguments=['--input_data', dataset.as_mount()],
                      compute_target=compute_target,
                      environment=tf_env,
                      distributed_job_config=MpiConfiguration(node_count=2))


run = experiment.submit(src)
print(run)
run.get_details()

run.wait_for_completion(show_output=True)