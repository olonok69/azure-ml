# Check core SDK version number
# https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/responsible-ai/visualize-upload-loan-decision/rai-loan-decision.ipynb

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

from fairlearn.reductions import GridSearch
from fairlearn.reductions import DemographicParity

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
# SHAP Tabular Explainer
from interpret.ext.blackbox import MimicExplainer
from interpret.ext.glassbox import LGBMExplainableModel
from utilities import fetch_census_dataset
from azureml.contrib.fairness import upload_dashboard_dictionary, download_dashboard_by_upload_id
from azureml.interpret import ExplanationClient
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




dataset = fetch_census_dataset()
X_raw, y = dataset['data'], dataset['target']
X_raw["race"].value_counts().to_dict()

sensitive_features = X_raw[['sex','race']]
le = LabelEncoder()
y = le.fit_transform(y)


X_train, X_test, y_train, y_test, sensitive_features_train, sensitive_features_test = \
    train_test_split(X_raw, y, sensitive_features,
                     test_size = 0.2, random_state=0, stratify=y)

# Work around indexing bug
X_train = X_train.reset_index(drop=True)
sensitive_features_train = sensitive_features_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
sensitive_features_test = sensitive_features_test.reset_index(drop=True)

# Training a fairness-unaware predictor

numeric_transformer = Pipeline(
    steps=[
        ("impute", SimpleImputer()),
        ("scaler", StandardScaler()),
    ]
)
categorical_transformer = Pipeline(
    [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, make_column_selector(dtype_exclude="category")),
        ("cat", categorical_transformer, make_column_selector(dtype_include="category")),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(solver="liblinear", fit_intercept=True),
        ),
    ]
)

model.fit(X_train, y_train)

#Generate model explanations

# Using SHAP KernelExplainer
# clf.steps[-1][1] returns the trained classification model
explainer = MimicExplainer(model.steps[-1][1], 
                           X_train,
                           LGBMExplainableModel,
                           features=X_raw.columns, 
                           classes=['Rejected', 'Approved'],
                           transformations=preprocessor)


# Explain the model based on a subset of 1000 rows
global_explanation = explainer.explain_global(X_test[:1000])

print(global_explanation.get_feature_importance_dict())


#Generate local explanations
# You can pass a specific data point or a group of data points to the explain_local function
# E.g., Explain the first data point in the test set
instance_num = 1
local_explanation = explainer.explain_local(X_test[:instance_num])

# Get the prediction for the first member of the test set and explain why model made that prediction
prediction_value = model.predict(X_test)[instance_num]

sorted_local_importance_values = local_explanation.get_ranked_local_values()[prediction_value]
sorted_local_importance_names = local_explanation.get_ranked_local_names()[prediction_value]

print('local importance values: {}'.format(sorted_local_importance_values))
print('local importance names: {}'.format(sorted_local_importance_names))

#Visualize model explanations
ExplanationDashboard(global_explanation, model, dataset=X_test[:1000], true_y=y_test[:1000])


# Assess model fairness

from raiwidgets import FairnessDashboard

y_pred = model.predict(X_test)

FairnessDashboard(sensitive_features=sensitive_features_test,
                  y_true=y_test,
                  y_pred=y_pred)

# Mitigation with Fairlearn (GridSearch)

# Fairlearn is not yet fully compatible with Pipelines, so we have to pass the estimator only
X_train_prep = preprocessor.transform(X_train).toarray()
X_test_prep = preprocessor.transform(X_test).toarray()

sweep = GridSearch(LogisticRegression(solver="liblinear", fit_intercept=True),
                   constraints=DemographicParity(),
                   grid_size=70)

sweep.fit(X_train_prep, y_train,
          sensitive_features=sensitive_features_train.sex)

predictors = sweep.predictors_

from fairlearn.metrics import demographic_parity_difference

accuracies, disparities = [], []

for predictor in predictors:
    y_pred = predictor.predict(X_train_prep)
    # accuracy_metric_frame = MetricFrame(accuracy_score, y_train, predictor.predict(X_train_prep), sensitive_features=sensitive_features_train.sex)
    # selection_rate_metric_frame = MetricFrame(selection_rate, y_train, predictor.predict(X_train_prep), sensitive_features=sensitive_features_train.sex)
    accuracies.append(accuracy_score(y_train, y_pred))
    disparities.append(demographic_parity_difference(y_train,
                                                     y_pred,
                                                     sensitive_features=sensitive_features_train.sex))
    
all_results = pd.DataFrame({"predictor": predictors, "accuracy": accuracies, "disparity": disparities})

all_models_dict = {"unmitigated": model.steps[-1][1]}
dominant_models_dict = {"unmitigated": model.steps[-1][1]}
base_name_format = "grid_{0}"
row_id = 0
for row in all_results.itertuples():
    model_name = base_name_format.format(row_id)
    all_models_dict[model_name] = row.predictor
    accuracy_for_lower_or_eq_disparity = all_results["accuracy"][all_results["disparity"] <= row.disparity]
    if row.accuracy >= accuracy_for_lower_or_eq_disparity.max():
        dominant_models_dict[model_name] = row.predictor
    row_id = row_id + 1


dashboard_all = {}
for name, predictor in all_models_dict.items():
    value = predictor.predict(X_test_prep)
    dashboard_all[name] = value
    
dominant_all = {}
for name, predictor in dominant_models_dict.items():
    dominant_all[name] = predictor.predict(X_test_prep)

FairnessDashboard(sensitive_features=sensitive_features_test, 
                  y_true=y_test,
                  y_pred=dominant_all)

# Registering models

os.makedirs('models', exist_ok=True)
def register_model(name, model):
    print("Registering ", name)
    model_path = "models/{0}.pkl".format(name)
    joblib.dump(value=model, filename=model_path)
    registered_model = Model.register(model_path=model_path,
                                      model_name=name,
                                      workspace=ws)
    print("Registered ", registered_model.id)
    return registered_model.id

model_name_id_mapping = dict()
for name, model in dominant_all.items():
    m_id = register_model(name, model)
    model_name_id_mapping[name] = m_id

dominant_all_ids = dict()
for name, y_pred in dominant_all.items():
    dominant_all_ids[model_name_id_mapping[name]] = y_pred

# Uploading a dashboard

sf = { 'sex': sensitive_features_test.sex, 'race': sensitive_features_test.race }

from fairlearn.metrics._group_metric_set import _create_group_metric_set

dash_dict_all = _create_group_metric_set(y_true=y_test,
                                         predictions=dominant_all_ids,
                                         sensitive_features=sf,
                                         prediction_type='binary_classification')


exp = Experiment(ws, 'responsible-ai-loan-decision')
print(exp)

run = exp.start_logging()
try:
    dashboard_title = "Upload MultiAsset from Grid Search with Census Data Notebook"
    upload_id = upload_dashboard_dictionary(run,
                                            dash_dict_all,
                                            dashboard_name=dashboard_title)
    print("\nUploaded to id: {0}\n".format(upload_id))

    downloaded_dict = download_dashboard_by_upload_id(run, upload_id)
finally:
    run.complete()


client = ExplanationClient.from_run(run)
client.upload_model_explanation(global_explanation, comment = "census data global explanation")
