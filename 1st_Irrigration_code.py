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
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()
print("Saved correlation heatmap as correlation_heatmap.png")

X = df[[c for c in df.columns if 'sensor' in c]]
y = df[[c for c in df.columns if 'parcel' in c]]

print("X shape:", X.shape)
print("y shape:", y.shape)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_split=4,
    min_samples_leaf=2, max_features='sqrt', random_state=42
)

model = MultiOutputClassifier(rf)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=y.columns))

print("Sum of parcels ON:")
print(df[['parcel_0', 'parcel_1', 'parcel_2']].sum())

conditions = {
    "Parcel 0 ON": df['parcel_0'],
    "Parcel 1 ON": df['parcel_1'],
    "Parcel 2 ON": df['parcel_2'],
    "Parcel 0 & 1 ON": df['parcel_0'] & df['parcel_1'],
    "Parcel 0 & 2 ON": df['parcel_0'] & df['parcel_2'],
    "Parcel 1 & 2 ON": df['parcel_1'] & df['parcel_2'],
    "All Parcels ON": df['parcel_0'] & df['parcel_1'] & df['parcel_2'],
}

fig, axs = plt.subplots(len(conditions), 1, figsize=(10, 15), sharex=True)

for ax, (title, cond) in zip(axs, conditions.items()):
    ax.step(df.index, cond.astype(int), where='post', linewidth=1, color='teal')
    ax.set_title(f"Sprinkler - {title}")
    ax.set_ylabel("Status")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['OFF', 'ON'])

axs[-1].set_xlabel("Time Index (Row Number)")

plt.tight_layout()
plt.savefig("sprinkler_conditions.png")
plt.close()
print("Saved sprinkler conditions plot as sprinkler_conditions.png")

plt.figure(figsize=(15, 5))
plt.step(df.index, df['parcel_0'], where='post', linewidth=2, label='Parcel 0 Pump', color='blue')
plt.step(df.index, df['parcel_1'], where='post', linewidth=2, label='Parcel 1 Pump', color='orange')
plt.step(df.index, df['parcel_2'], where='post', linewidth=2, label='Parcel 2 Pump', color='green')

plt.title("Pump Activity and Combined Farm Coverage")
plt.xlabel("Time Index (Row Number)")
plt.ylabel("Status")
plt.yticks([0, 1], ['OFF', 'ON'])
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("pump_activity.png")
plt.close()
print("Saved pump activity plot as pump_activity.png")

joblib.dump(model, "Farm_Irrigation_System.pkl")
print("Model saved as Farm_Irrigation_System.pkl")


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
