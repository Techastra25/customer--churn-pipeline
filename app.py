import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# 1.Preparing mock data (10,000 Records)
print("--- step 1: data generation and data preprocessing---")
np.random.seed(42)
num_records = 10000

X_raw = np.random.rand(num_records, 2) * 100 
y_raw = np.random.choice([0, 1], size=num_records, p=[0.8, 0.2]) 

df = pd.DataFrame(X_raw, columns=['tenure', 'monthly_charges'])
df['churn'] = y_raw

X = df[['tenure', 'monthly_charges']]
y = df['churn']

# 2. Training and Testing the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Correcting embalancy with the help of SMOTE  (4:1)
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# 4. Feature Scalling 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# 5. XGBoost Model Training 
print("\n--- step 2: XGBoost model training started---")
xgb_model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train_scaled, y_train_balanced)

# 6. Evaluation and Results
y_pred = xgb_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("\n---Final Model Results ---")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))
