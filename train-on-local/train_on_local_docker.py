# Check core SDK version number
import azureml.core
from azureml.core.workspace import Workspace
from azureml.core import Experiment ,Environment, ScriptRunConfig
from azureml.core.conda_dependencies import CondaDependencies
import subprocess
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

print("SDK version:", azureml.core.VERSION)
ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep='\n')

experiment_name = 'train-on-local'
exp = Experiment(workspace=ws, name=experiment_name)


# Editing a run configuration property on-fly.
docker_env = Environment("docker-env")

docker_env.python.user_managed_dependencies = False
docker_env.docker.enabled = True

# use the default CPU-based Docker image from Azure ML
print(docker_env.docker.base_image)

# Specify conda dependencies with scikit-learn
cd = CondaDependencies.create(conda_packages=['scikit-learn'])
# Specify conda dependencies with scikit-learn
docker_env.python.conda_dependencies = cd


src = ScriptRunConfig(source_directory='./', script='train.py', environment=docker_env)


src.run_config.environment = docker_env

# Check if Docker is installed and Linux containers are enabled
if subprocess.run("docker -v", shell=True).returncode == 0:
    out = subprocess.check_output("docker system info", shell=True).decode('ascii')
    if not "OSType: linux" in out:
        print("Switch Docker engine to use Linux containers.")
    else:
        run = exp.submit(src)
        run.wait_for_completion(show_output=True)
else:
    print("Docker engine is not installed.")
#run = exp.submit(src)

#run.wait_for_completion(show_output=True)


# Get all metris logged in the run
run.get_metrics()
metrics = run.get_metrics()

best_alpha = metrics['alpha'][np.argmin(metrics['mse'])]

print('When alpha is {1:0.2f}, we have min MSE {0:0.2f}.'.format(
    min(metrics['mse']), 
    best_alpha
))

plt.plot(metrics['alpha'], metrics['mse'], marker='o')
plt.ylabel("MSE")
plt.xlabel("Alpha")


run.get_file_names()

# Supply a model name, and the full path to the serialized model file.
model = run.register_model(model_name='best_ridge_model', model_path='./outputs/ridge_0.40.pkl')
print("Registered model:\n --> Name: {}\n --> Version: {}\n --> URL: {}".format(model.name, model.version, model.url))