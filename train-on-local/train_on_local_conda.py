# Check core SDK version number
import azureml.core
from azureml.core.workspace import Workspace
from azureml.core import Experiment ,Environment, ScriptRunConfig
from azureml.core.conda_dependencies import CondaDependencies

print("SDK version:", azureml.core.VERSION)
ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep='\n')

experiment_name = 'train-on-local'
exp = Experiment(workspace=ws, name=experiment_name)


# Editing a run configuration property on-fly.
system_managed_env = Environment("system-managed-env")

system_managed_env.python.user_managed_dependencies = False

# Specify conda dependencies with scikit-learn
cd = CondaDependencies.create(conda_packages=['scikit-learn'])
system_managed_env.python.conda_dependencies = cd

# You can choose a specific Python environment by pointing to a Python path 
#user_managed_env.python.interpreter_path = '/home/johndoe/miniconda3/envs/myenv/bin/python'

src = ScriptRunConfig(source_directory='./', script='train.py', environment=system_managed_env)
run = exp.submit(src)

run.wait_for_completion(show_output=True)
