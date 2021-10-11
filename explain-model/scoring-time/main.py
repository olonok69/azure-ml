# Check core SDK version number
# https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/explain-model/azure-integration/scoring-time/train-explain-model-locally-and-deploy.ipynb

import azureml.core
from azureml.telemetry import set_diagnostics_collection
from azureml.core.workspace import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import shutil
from azureml.core import Experiment, Environment, ScriptRunConfig
from azureml.core.dataset import Dataset
import urllib.request
from azureml.exceptions import UserErrorException
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
import pkg_resources
from azureml.interpret import ExplanationClient
from azureml.core.model import Model
import joblib
from raiwidgets import ExplanationDashboard
import requests
import json
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from interpret.ext.blackbox import TabularExplainer
from azureml.interpret.scoring.scoring_explainer import TreeScoringExplainer, save

from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.core.environment import Environment
from azureml.exceptions import WebserviceException

# make sure utils.py is in the same directory as this code


print("SDK version:", azureml.core.VERSION)

set_diagnostics_collection(send_diagnostics=True)

ws = Workspace.from_config()
print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep = '\n')


# Choose a name for your CPU cluster
cpu_cluster_name = "cpu-cluster"

# Verify that cluster does not exist already
try:
    cpu_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2',
                                                           max_nodes=4)
    cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config)

cpu_cluster.wait_for_completion(show_output=True)


experiment_name = 'explain_model_at_scoring_time'
experiment = Experiment(workspace=ws, name=experiment_name)
run = experiment.start_logging()


outdirname = 'dataset.6.21.19'
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve
import zipfile
zipfilename = outdirname + '.zip'
urlretrieve('https://publictestdatasets.blob.core.windows.net/data/' + zipfilename, zipfilename)
with zipfile.ZipFile(zipfilename, 'r') as unzip:
    unzip.extractall('.')
attritionData = pd.read_csv('./WA_Fn-UseC_-HR-Employee-Attrition.csv')


os.makedirs('./outputs', exist_ok=True)

# Dropping Employee count as all values are 1 and hence attrition is independent of this feature
attritionData = attritionData.drop(['EmployeeCount'], axis=1)
# Dropping Employee Number since it is merely an identifier
attritionData = attritionData.drop(['EmployeeNumber'], axis=1)
attritionData = attritionData.drop(['Over18'], axis=1)
# Since all values are 80
attritionData = attritionData.drop(['StandardHours'], axis=1)

# Converting target variables from string to numerical values
target_map = {'Yes': 1, 'No': 0}
attritionData["Attrition_numerical"] = attritionData["Attrition"].apply(lambda x: target_map[x])
target = attritionData["Attrition_numerical"]

attritionXData = attritionData.drop(['Attrition_numerical', 'Attrition'], axis=1)

# Creating dummy columns for each categorical feature
categorical = []
for col, value in attritionXData.iteritems():
    if value.dtype == 'object':
        categorical.append(col)

# Store the numerical columns in a list numerical
numerical = attritionXData.columns.difference(categorical)

# We create the preprocessing pipelines for both numeric and categorical data.
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

transformations = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical),
        ('cat', categorical_transformer, categorical)])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', transformations),
                      ('classifier', RandomForestClassifier())])

# Split data into train and test
x_train, x_test, y_train, y_test = train_test_split(attritionXData,
                                                    target,
                                                    test_size=0.2,
                                                    random_state=0,
                                                    stratify=target)

# Preprocess the data and fit the classification model
clf.fit(x_train, y_train)
model = clf.steps[-1][1]

model_file_name = 'log_reg.pkl'

# Save model in the outputs folder so it automatically get uploaded
with open(model_file_name, 'wb') as file:
    joblib.dump(value=clf, filename=os.path.join('./outputs/',
                                                 model_file_name))


# Explain predictions on your local machine
tabular_explainer = TabularExplainer(model, 
                                     initialization_examples=x_train, 
                                     features=attritionXData.columns, 
                                     classes=["Not leaving", "leaving"], 
                                     transformations=transformations)

# Explain overall model predictions (global explanation)
# Passing in test dataset for evaluation examples - note it must be a representative sample of the original data
# x_train can be passed as well, but with more examples explanations it will
# take longer although they may be more accurate
global_explanation = tabular_explainer.explain_global(x_test)


# ScoringExplainer
scoring_explainer = TreeScoringExplainer(tabular_explainer)
# Pickle scoring explainer locally
save(scoring_explainer, exist_ok=True)

# Register original model
run.upload_file('original_model.pkl', os.path.join('./outputs/', model_file_name))
original_model = run.register_model(model_name='local_deploy_model', 
                                    model_path='original_model.pkl')

# Register scoring explainer
run.upload_file('IBM_attrition_explainer.pkl', 'scoring_explainer.pkl')
scoring_explainer_model = run.register_model(model_name='IBM_attrition_explainer', model_path='IBM_attrition_explainer.pkl')

# Visualize

ExplanationDashboard(global_explanation, clf, dataset=x_test)

# Deploy
# azureml-defaults is required to host the model as a web service.
azureml_pip_packages = [
    'azureml-defaults', 'azureml-core', 'azureml-telemetry',
    'azureml-interpret', 'lightgbm', 'scikit-learn'
]
 
# Note: this is to pin the scikit-learn and pandas versions to be same as notebook.
# In production scenario user would choose their dependencies
import pkg_resources
available_packages = pkg_resources.working_set
sklearn_ver = None
pandas_ver = None
for dist in available_packages:
    if dist.key == 'scikit-learn':
        sklearn_ver = dist.version
    elif dist.key == 'pandas':
        pandas_ver = dist.version
sklearn_dep = 'scikit-learn'
pandas_dep = 'pandas'
if sklearn_ver:
    sklearn_dep = 'scikit-learn=={}'.format(sklearn_ver)
if pandas_ver:
    pandas_dep = 'pandas=={}'.format(pandas_ver)
# Specify CondaDependencies obj
# The CondaDependencies specifies the conda and pip packages that are installed in the environment
# the submitted job is run in.  Note the remote environment(s) needs to be similar to the local
# environment, otherwise if a model is trained or deployed in a different environment this can
# cause errors.  Please take extra care when specifying your dependencies in a production environment.
myenv = CondaDependencies.create(pip_packages=['pyyaml', sklearn_dep, pandas_dep] + azureml_pip_packages)

# with open("myenv.yml","w") as f:
#     f.write(myenv.serialize_to_string())

with open("myenv.yml","r") as f:
    print(f.read())


# Retrieve scoring explainer for deployment
scoring_explainer_model = Model(ws, 'IBM_attrition_explainer')


aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, 
                                               memory_gb=1, 
                                               tags={"data": "IBM_Attrition",  
                                                     "method" : "local_explanation"}, 
                                               description='Get local explanations for IBM Employee Attrition data')

myenv = Environment.from_conda_specification(name="myenv", file_path="myenv.yml")
inference_config = InferenceConfig(entry_script="score_local_explain.py", environment=myenv)

# Use configs and models generated above
service = Model.deploy(ws, 'model-scoring-deploy-local', [scoring_explainer_model, original_model], inference_config, aciconfig)
try:
    service.wait_for_deployment(show_output=True)
except WebserviceException as e:
    print(e.message)
    print(service.get_logs())
    raise

# Create data to test service with
sample_data = '{"Age":{"899":49},"BusinessTravel":{"899":"Travel_Rarely"},"DailyRate":{"899":1098},"Department":{"899":"Research & Development"},"DistanceFromHome":{"899":4},"Education":{"899":2},"EducationField":{"899":"Medical"},"EnvironmentSatisfaction":{"899":1},"Gender":{"899":"Male"},"HourlyRate":{"899":85},"JobInvolvement":{"899":2},"JobLevel":{"899":5},"JobRole":{"899":"Manager"},"JobSatisfaction":{"899":3},"MaritalStatus":{"899":"Married"},"MonthlyIncome":{"899":18711},"MonthlyRate":{"899":12124},"NumCompaniesWorked":{"899":2},"OverTime":{"899":"No"},"PercentSalaryHike":{"899":13},"PerformanceRating":{"899":3},"RelationshipSatisfaction":{"899":3},"StockOptionLevel":{"899":1},"TotalWorkingYears":{"899":23},"TrainingTimesLastYear":{"899":2},"WorkLifeBalance":{"899":4},"YearsAtCompany":{"899":1},"YearsInCurrentRole":{"899":0},"YearsSinceLastPromotion":{"899":0},"YearsWithCurrManager":{"899":0}}'


headers = {'Content-Type':'application/json'}

# Send request to service
print("POST to url", service.scoring_uri)
resp = requests.post(service.scoring_uri, sample_data, headers=headers)

# Can covert back to Python objects from json string if desired
print("prediction:", resp.text)
result = json.loads(resp.text)

# Plot the feature importance for the prediction
import numpy as np
import matplotlib.pyplot as plt; plt.rcdefaults()

labels = json.loads(sample_data)
labels = labels.keys()
objects = labels
y_pos = np.arange(len(objects))
performance = result["local_importance_values"][0][0]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
locs, labels = plt.xticks()
plt.setp(labels, rotation=90)
plt.ylabel('Feature impact - leaving vs not leaving')
plt.title('Local feature importance for prediction')

plt.show()


# service.delete()
