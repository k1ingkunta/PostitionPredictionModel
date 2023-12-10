import pandas as pd
import chardet
import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

file_path = 'RRRegular.csv'

# Detect the encoding of the CSV file
with open(file_path, 'rb') as file:
    result = chardet.detect(file.read())

# Read CSV file with pandas
data = pd.read_csv(file_path, encoding=result['encoding'])

# Assuming 'target_column' is the column you want to predict
target_column = 'Pos'
X = data.drop(target_column, axis=1)
y = data[target_column]

numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns

scaler = MinMaxScaler()

X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=24)

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Print the results
print("Random Forest Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


new_data_path = 'TESTOWNIK.csv'
new_data = pd.read_csv(new_data_path, delimiter=';')

target_column = 'Pos'
X_new = new_data.drop(target_column, axis=1)

y_pred_new = clf.predict(X_new)

print("Predictions for the new data:")
print(y_pred_new)
print("actuall positions: ")
print(new_data.head)

