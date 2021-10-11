import mlflow
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.exceptions import ComputeTargetException
import os, shutil
from azureml.core import Experiment, ScriptRunConfig, Environment, Workspace, Dataset
from azureml.widgets import RunDetails
import azureml.core
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import PipelineData, Pipeline
from azureml.pipeline.steps import ParallelRunConfig, ParallelRunStep
from azureml.pipeline.steps import PythonScriptStep
from azureml.core import Experiment
from azureml.core import Model
import pandas as pd
import numpy as np
import joblib, requests
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig
from azureml.core.runconfig import DEFAULT_CPU_IMAGE
from azureml.core.runconfig import CondaDependencies
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.pipeline.core.run import PipelineRun
from azureml.widgets import RunDetails

# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))


# Create an Azure ML experiment in your workspace
experiment = Experiment(workspace = ws, name = "diabetes-training")
run = experiment.start_logging()
print("Starting experiment:", experiment.name)

# load the diabetes dataset
print("Loading Data...")
diabetes = pd.read_csv('data/diabetes.csv')

# Separate features and labels
X, y = diabetes[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, diabetes['Diabetic'].values

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Train a decision tree model
print('Training a decision tree model')
model = DecisionTreeClassifier().fit(X_train, y_train)

# calculate accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
print('Accuracy:', acc)
run.log('Accuracy', np.float(acc))

# calculate AUC
y_scores = model.predict_proba(X_test)
auc = roc_auc_score(y_test,y_scores[:,1])
print('AUC: ' + str(auc))
run.log('AUC', np.float(auc))

# Save the trained model
model_file = 'diabetes_model.pkl'
joblib.dump(value=model, filename=model_file)
run.upload_file(name = 'outputs/' + model_file, path_or_stream = './' + model_file)

# Complete the run
run.complete()

# Register the model
run.register_model(model_path='outputs/diabetes_model.pkl', model_name='diabetes_model',
                   tags={'Training context':'Inline Training'},
                   properties={'AUC': run.get_metrics()['AUC'], 'Accuracy': run.get_metrics()['Accuracy']})

print('Model trained and registered.')

### Generate and Upload Batch Data

# Set default data store
ws.set_default_datastore('workspaceblobstore')
default_ds = ws.get_default_datastore()

# Enumerate all datastores, indicating which is the default
for ds_name in ws.datastores:
    print(ds_name, "- Default =", ds_name == default_ds.name)

# Load the diabetes data
diabetes = pd.read_csv('data/diabetes2.csv')
# Get a 100-item sample of the feature columns (not the diabetic label)
sample = diabetes[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].sample(n=100).values

# Create a folder
batch_folder = './batch-data'
os.makedirs(batch_folder, exist_ok=True)
print("Folder created!")

# Save each sample as a separate file
print("Saving files...")
for i in range(100):
    fname = str(i+1) + '.csv'
    sample[i].tofile(os.path.join(batch_folder, fname), sep=",")
print("files saved!")

# Upload the files to the default datastore
print("Uploading files to datastore...")
default_ds = ws.get_default_datastore()
default_ds.upload(src_dir="batch-data", target_path="batch-data", overwrite=True, show_progress=True)

# Register a dataset for the input data
batch_data_set = Dataset.File.from_files(path=(default_ds, 'batch-data/'), validate=False)
try:
    batch_data_set = batch_data_set.register(workspace=ws, 
                                             name='batch-data',
                                             description='batch data',
                                             create_new_version=True)
except Exception as ex:
    print(ex)

print("Done!")



# Choose a name for your CPU cluster
cpu_cluster_name = "cpu-cluster"

# Verify that cluster does not exist already
try:
    inference_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2',
                                                           max_nodes=2)
    inference_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config)

inference_cluster.wait_for_completion(show_output=True)


# Create a folder for the experiment files
experiment_folder = 'batch_pipeline'
os.makedirs(experiment_folder, exist_ok=True)

print(experiment_folder)

# Add dependencies required by the model
# For scikit-learn models, you need scikit-learn
# For parallel pipeline steps, you need azureml-core and azureml-dataprep[fuse]
cd = CondaDependencies.create(pip_packages=['scikit-learn','azureml-defaults','azureml-core','azureml-dataprep[fuse]'])

batch_env = Environment(name='batch_environment')
batch_env.python.conda_dependencies = cd
batch_env.docker.enabled = True
batch_env.docker.base_image = DEFAULT_CPU_IMAGE
print('Configuration ready.')


default_ds = ws.get_default_datastore()

output_dir = PipelineData(name='inferences', 
                          datastore=default_ds, 
                          output_path_on_compute='diabetes/results')

parallel_run_config = ParallelRunConfig(
    source_directory=experiment_folder,
    entry_script="batch_diabetes.py",
    mini_batch_size="5",
    error_threshold=10,
    output_action="append_row",
    environment=batch_env,
    compute_target=inference_cluster,
    node_count=2)

parallelrun_step = ParallelRunStep(
    name='batch-score-diabetes',
    parallel_run_config=parallel_run_config,
    inputs=[batch_data_set.as_named_input('diabetes_batch')],
    output=output_dir,
    arguments=[],
    allow_reuse=True
)

print('Steps defined')


pipeline = Pipeline(workspace=ws, steps=[parallelrun_step])
pipeline_run = Experiment(ws, 'batch_prediction_pipeline').submit(pipeline)
pipeline_run.wait_for_completion(show_output=True)

shutil.rmtree('diabetes-results', ignore_errors=True)

prediction_run = next(pipeline_run.get_children())
prediction_output = prediction_run.get_output_data('inferences')
prediction_output.download(local_path='diabetes-results')


for root, dirs, files in os.walk('diabetes-results'):
    for file in files:
        if file.endswith('parallel_run_step.txt'):
            result_file = os.path.join(root,file)

# cleanup output format
df = pd.read_csv(result_file, delimiter=":", header=None)
df.columns = ["File", "Prediction"]

# Display the first 20 results
print(df.head(20))

### Publish the Pipeline and use its REST Interface

published_pipeline = pipeline_run.publish_pipeline(
    name='Diabetes_Parallel_Batch_Pipeline', description='Batch scoring of diabetes data', version='1.0')

print(published_pipeline)

rest_endpoint = published_pipeline.endpoint
print(rest_endpoint)

interactive_auth = InteractiveLoginAuthentication()
auth_header = interactive_auth.get_authentication_header()
print('Authentication header ready.')



rest_endpoint = published_pipeline.endpoint
response = requests.post(rest_endpoint, 
                         headers=auth_header, 
                         json={"ExperimentName": "Batch_Pipeline_via_REST"})
run_id = response.json()["Id"]
print(run_id)

published_pipeline_run = PipelineRun(ws.experiments["Batch_Pipeline_via_REST"], run_id)
print(published_pipeline_run)

shutil.rmtree("diabetes-results", ignore_errors=True)

prediction_run = next(published_pipeline_run.get_children())
prediction_output = prediction_run.get_output_data("inferences")
prediction_output.download(local_path="diabetes-results")


for root, dirs, files in os.walk("diabetes-results"):
    for file in files:
        if file.endswith('parallel_run_step.txt'):
            result_file = os.path.join(root,file)

# cleanup output format
df = pd.read_csv(result_file, delimiter=":", header=None)
df.columns = ["File", "Prediction"]

# Display the first 20 results
print(df.head(20))