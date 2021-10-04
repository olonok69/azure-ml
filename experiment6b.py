import pandas as pd
import mlflow
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.exceptions import ComputeTargetException
import os, shutil
import requests
from azureml.core import Experiment, ScriptRunConfig, Environment, Workspace, Dataset, Model
from azureml.core.conda_dependencies import CondaDependencies
from azureml.widgets import RunDetails
import azureml.core
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core.run import PipelineRun
from azureml.pipeline.core import PipelineData, Pipeline
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.pipeline.core import ScheduleRecurrence, Schedule

# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))



# Get the most recent run of the pipeline
experiment_name = 'mslearn-diabetes-pipeline'
pipeline_experiment = ws.experiments.get(experiment_name)
pipeline_run = list(pipeline_experiment.get_runs())[0]

# Publish the pipeline from the run
published_pipeline = pipeline_run.publish_pipeline(
    name="diabetes-training-pipeline", description="Trains diabetes model", version="1.0")

print(published_pipeline)

rest_endpoint = published_pipeline.endpoint
print(rest_endpoint)

interactive_auth = InteractiveLoginAuthentication()
auth_header = interactive_auth.get_authentication_header()
print("Authentication header ready.")

rest_endpoint = published_pipeline.endpoint
response = requests.post(rest_endpoint, 
                         headers=auth_header, 
                         json={"ExperimentName": experiment_name})
run_id = response.json()["Id"]
print(run_id)

published_pipeline_run = PipelineRun(ws.experiments[experiment_name], run_id)
pipeline_run.wait_for_completion(show_output=True)

# Submit the Pipeline every Monday at 00:00 UTC
recurrence = ScheduleRecurrence(frequency="Week", interval=1, week_days=["Monday"], time_of_day="00:00")
weekly_schedule = Schedule.create(ws, name="weekly-diabetes-training", 
                                  description="Based on time",
                                  pipeline_id=published_pipeline.id, 
                                  experiment_name=experiment_name, 
                                  recurrence=recurrence)
print('Pipeline scheduled.')

schedules = Schedule.list(ws)
print(schedules)

pipeline_experiment = ws.experiments.get(experiment_name)
latest_run = list(pipeline_experiment.get_runs())[0]

latest_run.get_details()


"""
ss = Schedule.list(ws)
for s in ss:
    print(s)

schedule_id='e42aa4f5-fffa-4c68-891c-496a1f7ae46a'
def stop_by_schedule_id(ws, schedule_id):
    s = next(s for s in Schedule.list(ws) if s.id == schedule_id)
    s.disable()
    return s

stop_by_schedule_id(ws, schedule_id)

"""