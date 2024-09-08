import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

dataset = pd.read_csv('Crop and fertilizer dataset.csv')  


encoder = OneHotEncoder(handle_unknown='ignore')
X_encoded = encoder.fit_transform(dataset[['District_Name', 'Soil_color']])

# Combine with numerical features
X = dataset[['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temperature']]
print(X)

# Convert column names to strings to avoid type error
X.columns = X.columns.astype(str)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(dataset['Fertilizer'])
for code, class_name in enumerate(label_encoder.classes_):
    print(f"Code {code} is mapped to class '{class_name}'")


# Split the data into training, validation, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, shuffle=True)

# Initialize XGBoost Classifier with tuned parameters
model_crop = XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss',
    learning_rate=0.1,  # Adjust learning rate
    n_estimators=1000,   # Adjust number of trees  # Adjust minimum child weight
)

# Train the model
model_crop.fit(X_train, y_train)

# Evaluate on validation data
y_val_pred = model_crop.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred, average='weighted')
val_recall = recall_score(y_val, y_val_pred, average='weighted')
val_f1 = f1_score(y_val, y_val_pred, average='weighted')

print("Validation Accuracy:", val_accuracy)
print("Validation Precision:", val_precision)
print("Validation Recall:", val_recall)
print("Validation F1 Score:", val_f1)
print("\nValidation Classification Report:\n", classification_report(y_val, y_val_pred))

y_test_pred = model_crop.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

print("Test Accuracy:", test_accuracy)
print("Test Precision:", test_precision)
print("Test Recall:", test_recall)
print("Test F1 Score:", test_f1)
print("\nTest Classification Report:\n", classification_report(y_test, y_test_pred))
model_crop.get_booster().save_model('xgb_model.json')


joblib.dump(model_crop, 'xgb_model.joblib')
joblib.dump(encoder, 'onehotencoder.joblib')
joblib.dump(scaler, 'standardscaler.joblib')
