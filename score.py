import json
import numpy as np
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path('my_model')  # Replace 'my_model' with the name of your registered model
    # Load the model
    model = joblib.load(model_path)

def run(raw_data):
    data = json.loads(raw_data)['data']
    data = np.array(data)
    # Perform inference
    predictions = model.predict(data)
    return json.dumps(predictions.tolist())
