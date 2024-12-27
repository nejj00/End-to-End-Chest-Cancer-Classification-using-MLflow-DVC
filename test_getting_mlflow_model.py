import mlflow
from mlflow import MlflowClient

# Initialize MLflow client and model details
client = MlflowClient()
model_name = "VGG16Model"
alias_name = "champion"

mlflow.set_tracking_uri("http://localhost:5000")

model_version = client.get_model_version_by_alias(name=model_name, alias=alias_name)
if not model_version:
    raise ValueError(f"No versions found for model '{model_name}' with alias '{alias_name}'")
latest_version = model_version.version
print("Latest version: ", latest_version)

# Construct the model URI with the alias
model_uri = f"models:/{model_name}@{alias_name}"

# Load the model
loaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)

print(loaded_model)


"""
Before we load the model we have to check for the version.
We can keep the version in some files with meta data in the same location
where the model is.
When we are launching the prediction service for the first time we check if we have any model
in the folder and we load the latest one.
We just need a config file in which we set the alias of the model that will be for production.
Later when we change the production model, we either do the version before we do a prediction
or whenever the prediction service is restarted - this is yet to be decided.

"""