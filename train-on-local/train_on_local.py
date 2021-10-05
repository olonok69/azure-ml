# Check core SDK version number
import azureml.core
from azureml.core.workspace import Workspace
from azureml.core import Experiment ,Environment, ScriptRunConfig

print("SDK version:", azureml.core.VERSION)
ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep='\n')

experiment_name = 'train-on-local'
exp = Experiment(workspace=ws, name=experiment_name)

# View training and auxiliary scripts 
with open('./train.py', 'r') as f:
    print(f.read())

with open('./mylib.py', 'r') as f:
    print(f.read())

# Editing a run configuration property on-fly.
user_managed_env = Environment("user-managed-env")

user_managed_env.python.user_managed_dependencies = True

# You can choose a specific Python environment by pointing to a Python path 
#user_managed_env.python.interpreter_path = '/home/johndoe/miniconda3/envs/myenv/bin/python'

src = ScriptRunConfig(source_directory='./', script='train.py', environment=user_managed_env)
run = exp.submit(src)

run.wait_for_completion(show_output=True)
