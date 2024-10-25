# !pip install evidently
# !pip install xgboost
# !pip install openpyxl  # Required for Excel export

import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric
import os

# Load dataset
db = pd.read_csv("Credit_score_cleaned_data.csv")

# Drop Customer_ID as it's not useful for model training
db = db.drop(columns=['Customer_ID'])

# Identify categorical columns and apply Label Encoding
categorical_columns = db.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()
for col in categorical_columns:
    db[col] = label_encoder.fit_transform(db[col])

# Define features and target
target = 'Credit_Score'
features = [col for col in db.columns if col != target]

X = db[features]
y = db[target]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the XGBoost model
model = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42, n_estimators=50)
model.fit(X_train, y_train)

# Extract feature importance
feature_importances = model.get_booster().get_score(importance_type='gain')
fi_df = pd.DataFrame({
    'feature': list(feature_importances.keys()),
    'importance': list(feature_importances.values())
}).sort_values(by='importance', ascending=False).reset_index(drop=True)

# Prepare data for drift detection
reference_data = X_train.copy()
reference_data[target] = y_train
reference_data['prediction'] = model.predict(X_train)

current_data = X_test.copy()
current_data[target] = y_test
current_data['prediction'] = model.predict(current_data)

# Generate Drift Report for all columns
# Create ColumnDriftMetric for each feature in the dataset
custom_metrics = [ColumnDriftMetric(column_name=col, stattest='ks', stattest_threshold=0.05) 
                  for col in features]

# Add the target variable's drift test
custom_metrics.append(ColumnDriftMetric(column_name=target, stattest='chisquare', stattest_threshold=0.05))

# Run the report
custom_drift_report = Report(metrics=custom_metrics)
custom_drift_report.run(reference_data=reference_data, current_data=current_data)

# Extract drift scores for each column
drift_result = custom_drift_report.as_dict()
individual_drift_scores = {}
for metric in drift_result.get('metrics', []):
    if isinstance(metric, dict) and 'result' in metric:
        result = metric.get('result', {})
        column_name = result.get('column_name', '')
        drift_detected = result.get('drift_detected', False)
        stattest_name = result.get('stattest_name', 'unknown')
        p_value = result.get('p_value', None)
        drift_score = result.get('drift_score', None)
        if column_name:
            individual_drift_scores[column_name] = {
                'drift_detected': drift_detected,
                'p_value': p_value,
                'stattest_name': stattest_name,
                'drift_score': drift_score
            }

# Convert drift scores to DataFrame
drift_scores_df = pd.DataFrame([
    {
        'feature': col,
        'drift_detected': score_info['drift_detected'],
        'p_value': score_info['p_value'],
        'stattest': score_info['stattest_name'],
        'drift_score': score_info['drift_score']
    }
    for col, score_info in individual_drift_scores.items()
])

# Merge feature importance and drift scores into a single table
merged_df = pd.merge(fi_df, drift_scores_df, how='outer', on='feature')

# Save the merged table to Excel in OneDrive
one_drive_path = r"D:\Users\M.ARCEAGUIRRE\OneDrive - Farmers Insurance Group\MLMONITOR\DriftScoresandft.xlsx"
with pd.ExcelWriter(one_drive_path, engine='openpyxl') as writer:
    merged_df.to_excel(writer, sheet_name='Model_Metrics', index=False)

print(f"Feature importance and drift scores saved to {one_drive_path} as a single table.")
