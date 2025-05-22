# Customer_Segmentation![Screenshot (83)](https://github.com/user-attachments/assets/7eb2ba9b-6f3e-43f9-9fda-ca0013c7964b)
![Screenshot (84)](https://github.com/user-attachments/assets/bd72a5bb-205b-46c3-9129-2d189be6ad9d)
 Project Description: Customer Segmentation using K-Means
This project performs customer segmentation using unsupervised machine learning. The goal is to group mall customers into distinct clusters based on their demographic and behavioral attributes such as:

Age

Annual Income (in thousand dollars)

Spending Score (1–100)

🔧 Workflow Steps:
Data Loading: Reads customer data from a CSV file (Mall_Customers.csv).

Preprocessing:

Removes the CustomerID column.

Encodes the Gender column into numerical values.

Feature Scaling: Standardizes features to normalize the data.

Elbow Method: Determines the optimal number of clusters by plotting the inertia values for k = 1 to 10.

K-Means Clustering: Applies K-Means to segment customers into k = 5 clusters.

Visualization:

Plots the Elbow Curve.

Creates a scatter plot of clusters based on annual income and spending score.

Cluster Summary: Outputs average statistics for each cluster.

🧑‍💻 Languages & Libraries Used:
Programming Language:

Python 

Libraries & Frameworks:

pandas – for data manipulation

numpy – for numerical operations

matplotlib & seaborn – for data visualization

scikit-learn – for machine learning (K-Means and StandardScaler)

