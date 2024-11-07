'''diabetes_analysis.py'''


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib


data = pd.read_csv('diabetes.csv')
print(data.head())

print("\nMissing values in each column:")
print(data.isnull().sum())


sns.set(style='whitegrid')

fig, axes = plt.subplots(3, 3, figsize=(15, 10))
axes = axes.flatten()

for i, column in enumerate(data.columns):
    sns.histplot(data[column], bins=30, kde=True, ax=axes[i])
    axes[i].set_title(column)

plt.tight_layout()
plt.show()


correlation_matrix = data.corr()


plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()


X = data.drop(columns='Outcome')  
y = data['Outcome'] 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_model = LogisticRegression(max_iter=200)
logistic_model.fit(X_train_scaled, y_train)


y_pred_logistic = logistic_model.predict(X_test_scaled)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
print(f'\nLogistic Regression Accuracy: {accuracy_logistic:.2f}')
print("\nClassification Report:")
print(classification_report(y_test, y_pred_logistic))



joblib.dump(logistic_model, "logistic_model.pkl")
joblib.dump(scaler, "scaler.pkl")