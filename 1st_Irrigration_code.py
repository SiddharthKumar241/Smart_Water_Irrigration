import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
import joblib

print("Current working directory:", os.getcwd())

csv_path = r"d:\Python\Avi.py\Internship\irrigation_machine.csv"
df = pd.read_csv(csv_path)

if 'Unnamed: 0' in df.columns:
    df.drop('Unnamed: 0', axis=1, inplace=True)

print(df.info())
print(df.describe())

plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

X = df[[col for col in df.columns if 'sensor' in col]]
y = df[[col for col in df.columns if 'parcel' in col]]

print("X shape:", X.shape)
print("y shape:", y.shape)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42
)

model = MultiOutputClassifier(rf)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=y.columns))

joblib.dump(model, "multi_label_rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")



#prevoisu code
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.multioutput import MultiOutputClassifier
# from sklearn.metrics import classification_report
# from sklearn.preprocessing import MinMaxScaler
# import joblib
# print("Current working directory:", os.getcwd())
# csv_path = r"d:\Python\Avi.py\Internship\irrigation_machine.csv"
# df = pd.read_csv(csv_path)
# if 'Unnamed: 0' in df.columns:
#     df = df.drop('Unnamed: 0', axis=1)
# print(df.info())
# print(df.describe())
# plt.figure(figsize=(15, 10))
# sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
# plt.title("Feature Correlation Heatmap")
# plt.show()
# X = df[[col for col in df.columns if 'sensor' in col]]
# y = df[[col for col in df.columns if 'parcel' in col]]
# print("X shape:", X.shape)
# print("y shape:", y.shape)
# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# multi_rf = MultiOutputClassifier(rf)
# multi_rf.fit(X_train, y_train)
# y_pred = multi_rf.predict(X_test)
# print(classification_report(y_test, y_pred, target_names=y.columns))
# joblib.dump(multi_rf, "multi_label_rf_model.pkl")
# joblib.dump(scaler, "scaler.pkl")
