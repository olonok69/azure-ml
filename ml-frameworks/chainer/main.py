#https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/ml-frameworks/chainer/train-hyperparameter-tune-deploy-with-chainer/train-hyperparameter-tune-deploy-with-chainer.ipynb


# Check core SDK version number
import azureml.core
from azureml.telemetry import set_diagnostics_collection
from azureml.core.workspace import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
import numpy as np
import os

import shutil
from azureml.core import Experiment, Environment, ScriptRunConfig
from azureml.core.dataset import Dataset
import urllib.request
from azureml.exceptions import UserErrorException
# make sure utils.py is in the same directory as this code
from utils import load_data
from azureml.core.runconfig import DockerConfiguration, CondaDependencies
from azureml.train.hyperdrive.runconfig import HyperDriveConfig
from azureml.train.hyperdrive.sampling import RandomParameterSampling
from azureml.train.hyperdrive.policy import BanditPolicy
from azureml.train.hyperdrive.run import PrimaryMetricGoal
from azureml.train.hyperdrive.parameter_expressions import choice

from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig
from azureml.core.webservice import Webservice
from azureml.core.model import Model
from azureml.core.environment import Environment

import matplotlib.pyplot as plt
import urllib
import gzip
import struct
import requests


print("SDK version:", azureml.core.VERSION)

set_diagnostics_collection(send_diagnostics=True)

ws = Workspace.from_config()
print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep = '\n')

# choose a name for your cluster

cluster_name = "gpu-cluster"

try:
    compute_target = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing compute target')
except ComputeTargetException:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_NC6', 
                                                           max_nodes=1)

    # create the cluster
    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)

    compute_target.wait_for_completion(show_output=True)

# use get_status() to get a detailed status for the current cluster. 



project_folder = './chainer-mnist'
os.makedirs(project_folder, exist_ok=True)

shutil.copy('chainer_mnist.py', project_folder)
shutil.copy('chainer_score.py', project_folder)
shutil.copy('utils.py', project_folder)

# Create an experiment
experiment_name = 'chainer-mnist'
experiment = Experiment(ws, name=experiment_name)

chainer_env = Environment.from_conda_specification(name = 'chainer-5.1.0-gpu', file_path = './conda_dependencies.yml')

# Specify a GPU base image
chainer_env.docker.base_image = 'mcr.microsoft.com/azureml/intelmpi2018.3-cuda9.0-cudnn7-ubuntu16.04'

docker_config = DockerConfiguration(use_docker=True)

# Configure your training job
src = ScriptRunConfig(source_directory=project_folder,
                      script='chainer_mnist.py',
                      arguments=['--epochs', 10, '--batchsize', 128, '--output_dir', './outputs'],
                      compute_target=compute_target,
                      environment=chainer_env,
                      docker_runtime_config=docker_config)

# Tune model hyperparameters

param_sampling = RandomParameterSampling( {
    "--batchsize": choice(128, 256),
    "--epochs": choice(5, 10, 20, 40)
    }
)

hyperdrive_config = HyperDriveConfig(run_config=src,
                                     hyperparameter_sampling=param_sampling, 
                                     primary_metric_name='Accuracy',
                                     policy=BanditPolicy(evaluation_interval=1, slack_factor=0.1, delay_evaluation=3),
                                     primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                                     max_total_runs=8,
                                     max_concurrent_runs=4)

# start the HyperDrive run
hyperdrive_run = experiment.submit(hyperdrive_config)

hyperdrive_run.wait_for_completion(show_output=True)

assert(hyperdrive_run.get_status() == "Completed")
# Find and register best model
best_run = hyperdrive_run.get_best_run_by_primary_metric()
print(best_run.get_details()['runDefinition']['arguments'])


print(best_run.get_file_names())

model = best_run.register_model(model_name='chainer-dnn-mnist', model_path='outputs/model.npz')

# Deploy the model in ACI
shutil.copy('chainer_score.py', project_folder)

cd = CondaDependencies.create()
cd.add_conda_package('numpy')
cd.add_pip_package('chainer==5.1.0')
cd.add_pip_package("azureml-defaults")
cd.add_pip_package("azureml-opendatasets")
cd.save_to_file(base_directory='./', conda_file_path='myenv.yml')

print(cd.serialize_to_string())

myenv = Environment.from_conda_specification(name="myenv", file_path="myenv.yml")
inference_config = InferenceConfig(entry_script="chainer_score.py", environment=myenv,
                                   source_directory=project_folder)

aciconfig = AciWebservice.deploy_configuration(cpu_cores=1,
                                               auth_enabled=True, # this flag generates API keys to secure access
                                               memory_gb=1,
                                               tags={'name': 'mnist', 'framework': 'Chainer'},
                                               description='Chainer DNN with MNIST')

service = Model.deploy(workspace=ws,
                       name='chainer-mnist-1',
                       models=[model],
                       inference_config=inference_config,
                       deployment_config=aciconfig)
service.wait_for_deployment(True)
print(service.state)
print(service.scoring_uri)

# Test the deployed model

# #######################################################################
# retreive the API keys. two keys were generated.
key1, Key2 = service.get_keys()
print(key1)


# load compressed MNIST gz files and return numpy arrays
def load_data(filename, label=False):
    with gzip.open(filename) as gz:
        struct.unpack('I', gz.read(4))
        n_items = struct.unpack('>I', gz.read(4))
        if not label:
            n_rows = struct.unpack('>I', gz.read(4))[0]
            n_cols = struct.unpack('>I', gz.read(4))[0]
            res = np.frombuffer(gz.read(n_items[0] * n_rows * n_cols), dtype=np.uint8)
            res = res.reshape(n_items[0], n_rows * n_cols)
        else:
            res = np.frombuffer(gz.read(n_items[0]), dtype=np.uint8)
            res = res.reshape(n_items[0], 1)
    return res

data_folder = os.path.join(os.getcwd(), 'data/mnist')
os.makedirs(data_folder, exist_ok=True)

urllib.request.urlretrieve('https://azureopendatastorage.blob.core.windows.net/mnist/t10k-images-idx3-ubyte.gz',
                           filename=os.path.join(data_folder, 't10k-images-idx3-ubyte.gz'))
urllib.request.urlretrieve('https://azureopendatastorage.blob.core.windows.net/mnist/t10k-labels-idx1-ubyte.gz',
                           filename=os.path.join(data_folder, 't10k-labels-idx1-ubyte.gz'))

X_test = load_data(os.path.join(data_folder, 't10k-images-idx3-ubyte.gz'), False) / np.float32(255.0)
y_test = load_data(os.path.join(data_folder, 't10k-labels-idx1-ubyte.gz'), True).reshape(-1)

# send a random row from the test set to score
random_index = np.random.randint(0, len(X_test)-1)
input_data = "{\"data\": [" + str(random_index) + "]}"

headers = {'Content-Type':'application/json', 'Authorization': 'Bearer ' + key1}

# send sample to service for scoring
resp = requests.post(service.scoring_uri, input_data, headers=headers)

print("label:", y_test[random_index])
print("prediction:", resp.text[1])

plt.imshow(X_test[random_index].reshape((28,28)), cmap='gray')
plt.axis('off')
plt.show()


model = ws.models['chainer-dnn-mnist']
print("Model: {}, ID: {}".format('chainer-dnn-mnist', model.id))
       
webservice = ws.webservices['chainer-mnist-1']
print("Webservice: {}, scoring URI: {}".format('chainer-mnist-1', webservice.scoring_uri))


## Consume endpoint
""""
import urllib.request
import json
import os
import ssl

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

# Request data goes here
data = {
}

body = str.encode(json.dumps(data))

url = 'http://553d11b5-efd9-4056-ae85-74fcbdfa747c.westeurope.azurecontainer.io/score'
api_key = 'OY1BO4I10q15D1DazsGV5rjsyaV8TxTX' # Replace this with the API key for the web service
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(json.loads(error.read().decode("utf8", 'ignore')))

"""

