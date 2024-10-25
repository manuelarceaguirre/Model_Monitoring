# !pip install evidently
# !pip install xgboost
# !pip install openpyxl  # Required for Excel export

import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
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

# Plot feature importances (optional visualization)
plt.figure(figsize=(10, 6))
plt.barh(fi_df['feature'], fi_df['importance'])
plt.gca().invert_yaxis()
plt.xlabel('Importance')
plt.title('Feature Importances from XGBoost Model')
plt.show()

# Simulate data drift by modifying test data
current_data = X_test.copy()
current_data['Monthly_Inhand_Salary'] = current_data['Monthly_Inhand_Salary'] * np.random.uniform(0.7, 1.3, current_data.shape[0])
current_data['Credit_History_Age'] = current_data['Credit_History_Age'] * np.random.uniform(0.8, 1.2, current_data.shape[0])

# Prepare data for drift detection
reference_data = X_train.copy()
reference_data[target] = y_train
reference_data['prediction'] = model.predict(X_train)

current_data_no_target = current_data.copy()
current_data_no_target['prediction'] = model.predict(current_data_no_target)

current_data[target] = y_test
current_data['prediction'] = current_data_no_target['prediction']

# Generate Custom Data Drift Report
custom_metrics = [
    ColumnDriftMetric(column_name='Monthly_Inhand_Salary', stattest='ks', stattest_threshold=0.05),
    ColumnDriftMetric(column_name='Credit_History_Age', stattest='wasserstein', stattest_threshold=0.1),
    ColumnDriftMetric(column_name='Credit_Score', stattest='chisquare', stattest_threshold=0.05)
]

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
        'column': col,
        'drift_detected': score_info['drift_detected'],
        'p_value': score_info['p_value'],
        'stattest': score_info['stattest_name'],
        'drift_score': score_info['drift_score']
    }
    for col, score_info in individual_drift_scores.items()
])

# Save feature importances and drift scores to Excel in OneDrive
one_drive_path = r"C:\Users\YourUserName\OneDrive\Documents\model_metrics.xlsx"  # Replace with your OneDrive path
with pd.ExcelWriter(one_drive_path, engine='openpyxl') as writer:
    fi_df.to_excel(writer, sheet_name='Feature_Importance', index=False)
    drift_scores_df.to_excel(writer, sheet_name='Drift_Scores', index=False)

print(f"Feature importances and drift scores saved to {one_drive_path}")
