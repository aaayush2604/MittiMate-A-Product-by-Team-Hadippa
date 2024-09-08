# import pandas as pd
# import joblib
# import xgboost as xgb
# import numpy as np
# from xgboost import DMatrix
# from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# dataset=pd.read_csv('Crop and fertilizer dataset.csv')
# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(dataset['Fertilizer'])
# for index, label in enumerate(label_encoder.classes_):
#     print(f"{index}-{label}")

# classes = {index: label for index, label in enumerate(label_encoder.classes_)}

# scaler = joblib.load('standardscaler.joblib') 
# model_loaded = xgb.Booster() 
# model_loaded.load_model('xgb_model.json')  

# test_data = pd.read_csv('test.csv')

# X_test_numerical = test_data[['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temperature']]

# X_test_scaled = scaler.transform(X_test_numerical)

# dtest = DMatrix(X_test_scaled)

# y_test_pred = model_loaded.predict(dtest)

# predictionClasses=np.argmax(y_test_pred,axis=1)

# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(dataset['Fertilizer'])
# for index, label in enumerate(label_encoder.classes_):
#     print(f"{index}-{label}")

# classes = {index: label for index, label in enumerate(label_encoder.classes_)}

# predictions = [classes[index] for index in predictionClasses]

    

# # Convert predictions to class labels if needed
# # Assuming the model was trained for classification, and you need integer class labels
# # If you used probabilities, you might need to convert them to class labels based on a threshold.
# # For example, if the model is binary classification and outputs probabilities, you can do:
# # y_test_pred_class = (y_test_pred > 0.5).astype(int)

# print("Predictions:", predictions)

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from xgboost import DMatrix
import numpy as np

def predict_fertilizer(inputs):
    # Load the scaler and model
    scaler = joblib.load('standardscaler.joblib')
    model_loaded = xgb.Booster()
    model_loaded.load_model('xgb_model.json')

    # Prepare input data
    # Convert the single input array to a DataFrame
    input_df = pd.DataFrame([inputs], columns=['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temperature'])

    # Scale the input data
    X_test_scaled = scaler.transform(input_df)

    # Create DMatrix for prediction
    dtest = DMatrix(X_test_scaled)

    # Make prediction
    y_test_pred = model_loaded.predict(dtest)

    # Determine the predicted class index
    predictionClass = np.argmax(y_test_pred)

    # Load label encoder and create class mapping
    label_encoder = LabelEncoder()
    dataset = pd.read_csv('Crop and fertilizer dataset.csv')  # Assuming dataset.csv contains the label data
    label_encoder.fit(dataset['Fertilizer'])

    # Create a dictionary to map class indices to class names
    classes = {index: label for index, label in enumerate(label_encoder.classes_)}

    # Map the predicted class index to class name
    prediction = classes[predictionClass]

    return prediction



