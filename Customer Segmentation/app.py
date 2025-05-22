# =============================
# 📊 Customer Segmentation using K-Means
# =============================

# 📦 Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 🎨 Set Aesthetic Plot Style
sns.set(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 6)

# 🧪 Load Dataset
df = pd.read_csv("Mall_Customers.csv")
print("✅ Dataset Loaded Successfully!\n")
print(df.head())

# 🧹 Preprocess Data
# Drop unnecessary columns
if 'CustomerID' in df.columns:
    df.drop('CustomerID', axis=1, inplace=True)

# Encode Gender
if df['Gender'].dtype == 'object':
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# 🎯 Feature Selection
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

# 🔄 Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 📈 Elbow Method for Optimal Clusters
inertia = []
K = range(1, 11)
for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

# 📉 Plot Elbow Curve
plt.figure(figsize=(10, 6))
sns.lineplot(x=list(K), y=inertia, marker='o', linewidth=2.5)
plt.title('🔍 Elbow Method to Determine Optimal Clusters', fontsize=16)
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Inertia', fontsize=12)
plt.xticks(K)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# 🧠 Apply K-Means Clustering
optimal_k = 5  # You can choose based on the elbow plot
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 🎨 Visualize Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    palette='Set1',
    s=100,
    edgecolor='black',
    alpha=0.8
)
plt.title(f'📊 Customer Segmentation (k={optimal_k})', fontsize=16)
plt.xlabel('Annual Income (k$)', fontsize=12)
plt.ylabel('Spending Score (1-100)', fontsize=12)
plt.legend(title='Cluster')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# 📋 Cluster Summary
cluster_summary = df.groupby('Cluster')[features].mean().round(2)
print("\n📌 Cluster Summary:\n")
print(cluster_summary)
