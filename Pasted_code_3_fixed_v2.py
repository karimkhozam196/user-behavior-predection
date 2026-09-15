

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

df_raw = pd.read_csv("user_behavior_dataset.csv")
df3 = df_raw.copy()
df2 = df_raw.copy()
df3.head()

#1.pre1 handel missing data
#missing values
df3.isnull().sum()

print("Shape:", df3.shape)
print("\nColumns and dtypes:")
print(df3.dtypes)

chosen_feature = "App Usage Time (min/day)"
print("Chosen feature:", chosen_feature)
print(df3[chosen_feature].describe())

df2 = df3.copy()
df2.head()

df2.loc[0, chosen_feature] = None
df2.loc[1, chosen_feature] = None
df2.loc[200, chosen_feature] = None
df2.loc[100, chosen_feature] = None
df2.loc[10, chosen_feature] = None

print(df2.head())

print("\nMissing counts:")
print(df2.isnull().sum())

#vis missing using heatmap missing in each col
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
sns.heatmap(df2.isnull(), cbar=False , cmap='viridis')
plt.show()

df2['App Usage Time (min/day)'] = df2['App Usage Time (min/day)'].fillna(df2['App Usage Time (min/day)'].median())
#df2=df2.drop('Operating System',axis=1)
df2.isnull().sum() #count missing values
print(df2.head()) #replace missing value with median

df2.loc[3, 'Gender'] = None
df2.loc[10, 'Gender'] = None

df2['Gender'].isnull().sum()

df2[['User ID', 'Gender']].head(12)

df2['Gender'] = df2['Gender'].fillna(df2['Gender'].mode()[0]) #Fill missing Gender values with the mode
df2.isnull().sum()

df2[['User ID', 'Gender']].head(12)

#data cleaning
df2.duplicated().sum()

print(df2.head(10))

row_index = 5
duplicate_row = df2.iloc[row_index]
top = df2.iloc[:row_index+1]
bottom = df2.iloc[row_index+1:]
df2 = pd.concat([top, duplicate_row.to_frame().T, bottom], ignore_index=True)

print(df2.head(10))

df2.duplicated().sum()

df2 = df2.drop_duplicates() #remove duplicates

print(df2.head(10))

#feture eng

df2['Usage_Efficiency'] = df2['App Usage Time (min/day)'] / (df2['Screen On Time (hours/day)'] * 60)

print("feture eng",df2.head()) #new feature Usage_Efficiency

# feture scaling num diff scale normalization

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df2[['App Usage Time (min/day)',
    'Screen On Time (hours/day)',
    'Battery Drain (mAh/day)',
    'Data Usage (MB/day)',
    'Number of Apps Installed',
    'Age']] = scaler.fit_transform(df2[['App Usage Time (min/day)',
    'Screen On Time (hours/day)',
    'Battery Drain (mAh/day)',
    'Data Usage (MB/day)',
    'Number of Apps Installed',
    'Age']])
df2.head()

#catagorical to numrical

df2.dtypes

from sklearn.preprocessing import LabelEncoder
encode=LabelEncoder()
df2[str('Gender')] = encode.fit_transform(df2['Gender']).tolist()
df2[str('Operating System')] = encode.fit_transform(df2['Operating System']).tolist()
df2.head()

df2.drop(['Device Model'],axis=1,inplace=True)
df2.head()

#outliers

df2.describe()

df2['App Usage Time (min/day)'].describe()

p1=sb.boxplot(x=df2['App Usage Time (min/day)'])

df2.loc[0, 'App Usage Time (min/day)'] = 20
df2.loc[3, 'App Usage Time (min/day)'] = 40

p1=sb.boxplot(x=df2['App Usage Time (min/day)'])

df2 = df2[(df2['App Usage Time (min/day)'] >= -1) &
          (df2['App Usage Time (min/day)'] <= 1)]

outliers=['App Usage Time (min/day)']

df4=df2[df2['App Usage Time (min/day)']<=4]
p1=sb.boxplot(x=df4['App Usage Time (min/day)'])

#correlation matrix relation between variables

df2.corr()

plt.figure(figsize=(10,4))
sb.heatmap(df2.corr(),annot=True,annot_kws=dict(size=10),vmin=-1,vmax=1)
plt.show()

# -------------------- Feature Importance from Correlation --------------------
target = "Screen On Time (hours/day)"

corr_target = df2.corr()[target].sort_values(ascending=False)

print("\nCorrelation with Target (sorted):")
print(corr_target)

plt.figure(figsize=(8,4))
corr_target.plot(kind='bar')
plt.title("Feature Importance Based on Correlation")
plt.ylabel("Correlation Value")
plt.show()

#filtration

f_corr=df2.corr()[(df2.corr()>=0.5)|(df2.corr()<=-0.5)]
print(f_corr)

import pandas as pd
df=df3.copy()
# أعمدة هنعمل لها binning إلى 3 فئات (0,1,2)
cols_3bins = [
    "App Usage Time (min/day)",
    "Screen On Time (hours/day)",
    "Battery Drain (mAh/day)",
    "Number of Apps Installed",
    "Data Usage (MB/day)"
]

for col in cols_3bins:
    df[col + "_bin"] = pd.qcut(df[col], q=3, labels=[0, 1, 2])

# عمود Age نخليه 4 فئات (0..3)
df["Age_bin"] = pd.qcut(df2["Age"], q=4, labels=[0, 1, 2, 3])

print(df.head())

#feature selection chi square

df=df[
    [
        "App Usage Time (min/day)_bin",
        "Screen On Time (hours/day)_bin",
        "Battery Drain (mAh/day)_bin",
        "Number of Apps Installed_bin",
        "Data Usage (MB/day)_bin",
        "Age_bin",
        "Gender",
        #"Device Model",
        "Operating System",
        #"User Behavior Class"
    ]
]
#df=df[['Gender', 'Operating System']]
for col in df.columns:
  le=LabelEncoder()
  df[col]=df[col].fillna(df[col].mode()[0])
df.head()

#label encoding
from sklearn.preprocessing import LabelEncoder
for col in df.columns:
   le=LabelEncoder()
   df[col]=le.fit_transform(df[col])
df.head()

from sklearn.feature_selection import chi2
x = df.drop(columns=['Operating System'])
y = df['Operating System']

chi_scores = chi2(x,y)

#chi values higer vale higer importance

chi_values = pd.Series(chi_scores[0], index=x.columns)
chi_values.sort_values(ascending=False, inplace=True)
chi_values.plot.bar()

#higer p lower imp

p_values = pd.Series(chi_scores[1], index=x.columns)
p_values.sort_values(ascending=False, inplace=True)
p_values.plot.bar()

chi_values = pd.Series(chi_scores[0], index=x.columns)
p_values = pd.Series(chi_scores[1], index=x.columns)

feature_rank = pd.DataFrame({
    "Chi-Square": chi_values,
    "p-value": p_values
}).sort_values(by="Chi-Square", ascending=False)

print("\nTop Features Based on Chi-Square:\n")
print(feature_rank)

#kmeans clustring
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

km_df=df3.copy()

Features = km_df.drop(columns=["Device Model", "User ID", "User Behavior Class"])
target = km_df[["User Behavior Class"]]

Gender_encoder = LabelEncoder()
OS_encoder = LabelEncoder()

Features["Gender"] = Gender_encoder.fit_transform(km_df["Gender"])
Features["Operating System"] = OS_encoder.fit_transform(km_df["Operating System"])

scaler = StandardScaler()
Features_scaled = scaler.fit_transform(Features)

inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(Features_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(k_range, inertia, marker='o')
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for K-Means")
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(Features_scaled)
df["Cluster"] = clusters

pca = PCA(n_components=2)
Features_pca = pca.fit_transform(Features_scaled)

plt.scatter(Features_pca[:, 0], Features_pca[:, 1], c=df["Cluster"])
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("K-Means Clustering Visualization")
plt.show()

cluster_summary = df.groupby("Cluster").mean(numeric_only=True)
print(cluster_summary)

# =======================
# DECISION TREE MODEL
# =======================

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import matplotlib.pyplot as plt

# Copy raw data
dt_df = df3.copy()

# Fix column names
dt_df.columns = [col.strip() for col in dt_df.columns]

# Select features
selected_features = [
    'User ID',
    'Device Model',
    'Operating System',
    'App Usage Time (min/day)',
    'Screen On Time (hours/day)',
    'Battery Drain (mAh/day)',
    'Number of Apps Installed',
    'Data Usage (MB/day)',
    'Age',
    'Gender'
]

X = dt_df[selected_features].copy()
y = dt_df['User Behavior Class'].copy()

# Encode categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns
le_dict = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# Model
dt_model = DecisionTreeClassifier(max_depth=3)
dt_model.fit(X_train, y_train)

# Prediction
y_pred = dt_model.predict(X_test)
y_dt_pred = y_pred


# Accuracy for summary table
decision_tree_acc = metrics.accuracy_score(y_test, y_pred)
print("Decision Tree Accuracy:", decision_tree_acc)

# Plot the tree
plt.figure(figsize=(20, 10))
plot_tree(dt_model, filled=True, feature_names=selected_features)
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
df11=df3.copy()
actual = np.random.binomial(1, 0.9, size= 1000)
predicted = np.random.binomial(1, 0.9, size= 1000)

confusion_metrics= metrics.confusion_matrix(actual, predicted)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix= confusion_metrics, display_labels=[0, 1])
cm_display.plot()
plt.show()

df11_numeric = df11.select_dtypes(include=["int64","float64"])
print(df11_numeric.groupby(df11["User Behavior Class"]).mean())
df11_numeric.groupby("User Behavior Class").mean()

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# 1) Load clean dataset
df_nb = df3.copy()    # استخدم الداتا الخام

# 2) Select only independent features
X = df_nb[["Gender", "Operating System"]].copy()
y = df_nb["User Behavior Class"].copy()

# 3) Encode categorical values
le_gender = LabelEncoder()
X["Gender"] = le_gender.fit_transform(X["Gender"])

le_os = LabelEncoder()
X["Operating System"] = le_os.fit_transform(X["Operating System"])

# 4) Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# 5) Train Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

# 6) Predictions
y_pred = nb.predict(X_test)

# 7) Evaluation
print(classification_report(y_test, y_pred))

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# ===============================
# 1) COPY DATA
# ===============================
naive_df = df3.copy()

# ===============================
# 2) SPLIT FEATURES & TARGET
# ===============================
X = naive_df.drop(columns=["User Behavior Class"])
y = naive_df["User Behavior Class"]

# ===============================
# 3) LABEL ENCODING
# ===============================
for col in X.select_dtypes(include=["object"]).columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# ===============================
# 4) TRAIN / TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# ===============================
# 5) TRAIN MODEL
# ===============================
naive_model = GaussianNB()
naive_model.fit(X_train, y_train)

# ===============================
# 6) PREDICT
# ===============================
y_pred = naive_model.predict(X_test)

y_nb_pred=y_pred

# ===============================
# 7) ACCURACY (IMPORTANT)
# ===============================
naive_acc = accuracy_score(y_test, y_pred)
print("Naive Bayes Accuracy:", naive_acc)
# 8) Report
print(classification_report(y_test, y_pred))

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# 1) Load clean dataset
df_nb = df3.copy()    # استخدم الداتا الخام

# 2) Select only independent features
X = df_nb[["Gender", "Operating System"]].copy()
y = df_nb["User Behavior Class"].copy()

# 3) Encode categorical values
le_gender = LabelEncoder()
X["Gender"] = le_gender.fit_transform(X["Gender"])

le_os = LabelEncoder()
X["Operating System"] = le_os.fit_transform(X["Operating System"])

# 4) Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# 5) Train Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

# 6) Predictions
y_pred = nb.predict(X_test)

# 7) Evaluation
print(classification_report(y_test, y_pred))

# ---------------- LOGISTIC REGRESSION ---------------- #

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Copy original dataset
df_log = df3.copy()

# Features
features = [
    'User ID',
    'Device Model',
    'Operating System',
    'App Usage Time (min/day)',
    'Screen On Time (hours/day)',
    'Battery Drain (mAh/day)',
    'Number of Apps Installed',
    'Data Usage (MB/day)',
    'Age',
    'Gender'
]

X = df_log[features].copy()
y = df_log["User Behavior Class"].copy()

# Encode categorical columns
for col in X.select_dtypes(include="object").columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Model
logistic_model = LogisticRegression(max_iter=3000)
logistic_model.fit(X_train, y_train)

# Predictions
y_pred = logistic_model.predict(X_test)
y_log_pred=y_pred

# Accuracy
logistic_acc = accuracy_score(y_test, y_pred)
print("Logistic Regression Accuracy:", logistic_acc)
print("🔵 Logistic Regression Report:")
print(classification_report(y_test, y_pred))

# Commented out IPython magic to ensure Python compatibility.
# ========================================================
# 1) IMPORT LIBRARIES
# ========================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
# %matplotlib inline





neural_df = df3.copy()

print("Dataset loaded successfully!")
print(neural_df.head())


# 3) PREPROCESSING (Encoding + Scaling)


# Split X and y
X = neural_df.drop(columns=["User Behavior Class"])
y = neural_df["User Behavior Class"]

# FIX: Convert labels
y = y - 1

# Label Encoding for categorical columns
encoders = {}
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Scaling numeric values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

num_features = X_scaled.shape[1]
print("\nNumber of features:", num_features)


# ========================================================
# 4) TRAIN / TEST SPLIT + Convert to Tensors
# ========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # shape = (N,1,F)
X_test  = torch.tensor(X_test , dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(np.array(y_train), dtype=torch.long)
y_test  = torch.tensor(np.array(y_test) , dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test , y_test ), batch_size=16)


# ========================================================
# 5) CNN MODEL (1D)
# ========================================================
class ConvolutionalNetwork(nn.Module):
    def __init__(self, num_features, num_classes=5):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3)

        conv_output = num_features - 4  # after two 3-kernel conv layers

        self.fc1 = nn.Linear(conv_output * 32, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.relu(self.conv2(X))
        X = X.view(X.size(0), -1)
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        return F.log_softmax(X, dim=1)


# ========================================================
# 6) INIT MODEL
# ========================================================
torch.manual_seed(7)
model = ConvolutionalNetwork(num_features)
print(model)


# ========================================================
# 7) LOSS + OPTIMIZER
# ========================================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ========================================================
# 8) TRAINING LOOP
# ========================================================
import time
start_time = time.time()

epochs = 5
train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
    trn_corr = 0
    tst_corr = 0

    # -------- TRAIN --------
    for b, (X_batch, y_batch) in enumerate(train_loader):
        b += 1

        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)

        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_batch).sum()
        trn_corr += batch_corr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if b % 50 == 0:
            print(f"Epoch: {i} | Batch: {b} | Loss: {loss.item():.4f}")

    train_losses.append(loss.item())
    train_correct.append(trn_corr)

    # -------- TEST --------
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_val = model(X_batch)
            predicted = torch.max(y_val.data, 1)[1]
            tst_corr += (predicted == y_batch).sum()

    loss = criterion(y_val, y_batch)
    test_losses.append(loss.item())
    test_correct.append(tst_corr)


end_time = time.time()
print(f"\nTraining Took: {(end_time - start_time)/60:.2f} minutes")


# ========================================================
# 9) FINAL ACCURACY
# ========================================================
accuracy = tst_corr.item() / len(y_test)
print(f"\n FINAL CNN Accuracy = {accuracy}")


# ========================================================
# 10) CLASSIFICATION REPORT
# ========================================================
y_pred_final = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        output = model(X_batch)
        pred = torch.max(output.data, 1)[1]
        y_pred_final.extend(pred.tolist())
        y_cnn_pred = y_pred_final

print("CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred_final))

print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred_final))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("\n==================== MODEL SUMMARY ====================\n")

models = [
    ("Decision Tree", y_dt_pred, decision_tree_acc),
    ("Naive Bayes", y_nb_pred, naive_acc),
    ("Logistic Regression", y_log_pred, logistic_acc),
    ("1D CNN Model", y_cnn_pred, accuracy)
]

for name, pred, acc in models:
    print(f"{name}:")
    print(f"Accuracy  : {acc}")
    print(f"Precision : {precision_score(y_test, pred, average='weighted')}")
    print(f"Recall    : {recall_score(y_test, pred, average='weighted')}")
    print(f"F1-Score  : {f1_score(y_test, pred, average='weighted')}")
    print("-------------------------------------------------------")

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# =========================================================
# Load dataset
# =========================================================
ran_df= df3.copy()

# =========================================================
# Preprocessing
# =========================================================

# Split X and y
X = ran_df.drop(columns=["User Behavior Class"])
y = ran_df["User Behavior Class"]

# Fix labels from 1..5 → 0..4
y = y - 1

# Label Encoding for categorical columns
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# =========================================================
# Run Naive Bayes 20 times with random splits
# =========================================================
accuracies = []

for seed in range(1, 21):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed
    )

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    accuracies.append(acc)
    print(f"Run {seed:02d}  →  Accuracy = {acc}")

# =========================================================
# Final Results
# =========================================================
print("\n================ FINAL SUMMARY ================\n")
print(f"All Accuracies: {accuracies}")
print(f"\nAverage Accuracy : {np.mean(accuracies)}")
print(f"Max Accuracy     : {np.max(accuracies)}")
print(f"Min Accuracy     : {np.min(accuracies)}")
print("\n================================================")

import joblib
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# =============== LOAD + PREPROCESS ===============
df = df3.copy()

X = df.drop(columns=['User Behavior Class','User ID','Device Model'])
y = df["User Behavior Class"]

# Encode categorical columns
encoders = {}
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Save encoders
joblib.dump(encoders, "encoders.joblib")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# =============== TRAIN MODEL ===============
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Save model
joblib.dump(nb_model, "naive_bayes_model.joblib")

print("Model Saved Successfully!")

import joblib
import pandas as pd

# Load trained model + encoders
nb_model = joblib.load("naive_bayes_model.joblib")
encoders = joblib.load("encoders.joblib")

# Features needed in exact order
features = [
    #'User ID',
    #'Device Model',
    'Operating System',
    'App Usage Time (min/day)',
    'Screen On Time (hours/day)',
    'Battery Drain (mAh/day)',
    'Number of Apps Installed',
    'Data Usage (MB/day)',
    'Age',
    'Gender'
]

# Mapping class numbers → labels
class_names = {
    1: "Normal User",
    2: "Moderate User",
    3: "Heavy User",
    4: "Addicted User",
    5: "Super Heavy User"
}

# =============== PREDICTION FUNCTION ===============
def predict_user_dict(user_input_dict):

    # Convert user input to DataFrame row
    df_input = pd.DataFrame([user_input_dict], columns=features)

    # Apply encoders for categorical data
    for col in df_input.select_dtypes(include="object").columns:
        df_input[col] = encoders[col].transform(df_input[col])

    # Predict class
    prediction = nb_model.predict(df_input)[0]

    # Convert class number → text
    return class_names[prediction]

#User Behavior Class Criteria (Approximate Guidelines)
#1️⃣ Normal User

#App Usage Time: 0–120 min/day

#Screen On Time: 0–2 hours/day

#Battery Drain: 0–1500 mAh/day

#Installed Apps: 0–40 apps

#Data Usage: 0–500 MB/day
#➡️ Represents light and casual smartphone usage.

#2️⃣ Moderate User

#App Usage Time: 120–240 min/day

#Screen On Time: 2–4 hours/day

#Battery Drain: 1500–2500 mAh/day

#Installed Apps: 40–60 apps

#Data Usage: 500–1000 MB/day
#➡️ Uses the phone regularly but not excessively.

#3️⃣ Heavy User

#App Usage Time: 240–360 min/day

#Screen On Time: 4–6 hours/day

#Battery Drain: 2500–3500 mAh/day

#Installed Apps: 60–80 apps

#Data Usage: 1000–1500 MB/day
#➡️ Active user with high screen interaction.

#4️⃣ Addicted User

#App Usage Time: 360–480 min/day

#Screen On Time: 6–8 hours/day

#Battery Drain: 3500–4500 mAh/day

#Installed Apps: 80–100 apps

#Data Usage: 1500–2500 MB/day
#➡️ Excessive daily usage, strong dependency signs.

#5️⃣ Super Heavy User

#App Usage Time: 480+ min/day

#Screen On Time: 8+ hours/day

#Battery Drain: 4500+ mAh/day

#Installed Apps: 100+ apps

#Data Usage: 2500+ MB/day
#➡️ Extremely high engagement, constant device use.

#example
""" sample_user = {
    'Operating System': "Android",
    'App Usage Time (min/day)': 240,
    'Screen On Time (hours/day)': 6,
    'Battery Drain (mAh/day)': 3500,
    'Number of Apps Installed': 80,
    'Data Usage (MB/day)': 1500,
    'Age': 23,
    'Gender': "Male"
}

print(predict_user_dict(sample_user)) """



import gradio as gr

def make_prediction(
    operating_system,
    app_usage_time,
    screen_on_time,
    battery_drain,
    number_of_apps,
    data_usage,
    age,
    gender
):
    try:
        # Create input dictionary
        user_input_dict = {
            'Operating System': operating_system,
            'App Usage Time (min/day)': float(app_usage_time),
            'Screen On Time (hours/day)': float(screen_on_time),
            'Battery Drain (mAh/day)': float(battery_drain),
            'Number of Apps Installed': int(number_of_apps),
            'Data Usage (MB/day)': float(data_usage),
            'Age': int(age),
            'Gender': gender
        }

        # Create DataFrame
        df_input = pd.DataFrame([user_input_dict])

        # Encode categorical columns using saved encoders
        for col in df_input.select_dtypes(include=["object"]).columns:
            if col in encoders:
                df_input[col] = encoders[col].transform(df_input[col])

        # Prediction
        prediction = nb_model.predict(df_input)[0]

        return class_names[prediction]

    except Exception as e:
        return f"Error: {str(e)}"


iface = gr.Interface(
    fn=make_prediction,
    inputs=[
        gr.Dropdown(
            choices=["Android", "iOS", "Windows"],
            label="Operating System"
        ),
        gr.Number(
            label="App Usage Time (min/day)"
        ),
        gr.Number(
            label="Screen On Time (hours/day)"
        ),
        gr.Number(
            label="Battery Drain (mAh/day)"
        ),
        gr.Number(
            label="Number of Apps Installed"
        ),
        gr.Number(
            label="Data Usage (MB/day)"
        ),
        gr.Number(
            label="Age"
        ),
        gr.Dropdown(
            choices=["Male", "Female"],
            label="Gender"
        )
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="User Behavior Prediction App",
    description="Enter the user's information to predict the User Behavior Class."
)

iface.launch()