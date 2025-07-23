# Mall Customer Segmentation 🛍️👥

## Project Overview

This project focuses on performing **customer segmentation** for a mall's customer base using unsupervised machine learning techniques. By grouping customers into distinct segments based on their demographic and spending habits, the mall can gain valuable insights for targeted marketing strategies, personalized promotions, and improved customer experience.

---

## Dataset

The dataset used is assumed to be a customer dataset (e.g., `Mall_Customers.csv` or similar), containing information about mall customers. Key features typically include:

* `CustomerID`: Unique identifier for each customer.
* `Gender`: Gender of the customer.
* `Age`: Age of the customer.
* `Annual Income (k$)`: Customer's annual income in thousands of dollars.
* `Spending Score (1-100)`: A score assigned by the mall based on customer behavior and spending habits.

---

## Project Goals

* **Data Understanding:** Explore the distributions and characteristics of customer data.
* **Feature Selection:** Identify relevant features for segmentation.
* **Data Preprocessing:** Prepare numerical features for clustering algorithms.
* **Clustering:** Apply an unsupervised learning algorithm to group customers.
* **Optimal Cluster Determination:** Use appropriate methods to find the best number of segments.
* **Segment Analysis:** Interpret the characteristics of each identified customer segment.
* **Deployment Preparation:** Prepare the trained model for serving cluster predictions via an API.

---

## Methodology and Steps Taken

The project followed a structured unsupervised machine learning pipeline:

### Data Loading & Initial Inspection

* Loaded the customer dataset into a Pandas DataFrame.
* Performed initial checks (`df.head()`, `df.info()`, `df.describe()`) to understand data types, identify missing values, and get basic statistics.

### Exploratory Data Analysis (EDA) & Visualization

* **Univariate Analysis:** Visualized the distribution of individual features (`Age`, `Annual Income`, `Spending Score`) using histograms to understand their spread and patterns.
* **Bivariate Analysis:**
    * Used **2D Scatter Plots** (specifically, a `pairplot`) to visualize the relationships between pairs of key numerical features (`Age`, `Annual Income`, `Spending Score`). This helped in visually identifying potential clusters or correlations.
    * Integrated `Gender` into the pair plot using `hue` to observe if gender played a role in the distribution of these features, with numerical gender labels (0, 1) mapped to descriptive strings ('Female', 'Male') for clarity in the legend.
* **Correlation Matrix (Heatmap):** Generated a heatmap to quantify the linear relationships between all numerical features, providing a quick overview of feature interdependencies.

### Data Preprocessing for Clustering

* **Feature Selection:** Selected relevant numerical features (`Age`, `Annual Income`, `Spending Score`) as input for the clustering algorithm.
* **Feature Scaling:** Applied **`StandardScaler`** to these numerical features. This is a critical step for distance-based algorithms like K-Means, as it ensures all features contribute equally to distance calculations, preventing features with larger ranges from dominating.

### K-Means Clustering & Optimal K Determination

* **Elbow Method:** Employed the **Elbow Method** using the **Within-Cluster Sum of Squares (WCSS)**. The K-Means algorithm was run for a range of cluster numbers (K=1 to 10), and the WCSS for each K was plotted. The "elbow point" on this graph indicated the optimal number of clusters where the reduction in WCSS significantly diminishes.
* **Model Training:** Trained the final K-Means model using the optimal number of clusters determined by the Elbow Method.
* **Cluster Assignment:** Assigned a cluster label to each customer in the dataset based on the trained K-Means model.

### Segment Analysis

* Calculated **cluster statistics** (e.g., mean Age, Annual Income, Spending Score for each cluster) to understand the characteristics of each customer segment. This step is vital for interpreting the meaning of each cluster and formulating targeted strategies.
* Presented cluster statistics in a clear, formatted table using the `tabulate` library.

---

## Results

* **Optimal Number of Clusters (K):** 3
* **Customer Segments:** The analysis revealed 3 distinct customer segments, each with unique characteristics based on Age, Annual Income, and Spending Score.
    * **Segment 0 (Targetable/Balanced):** Typically comprises middle-aged customers with average annual income and moderate spending scores. This segment represents a stable customer base.
    * **Segment 1 (Spendthrifts/High-Value):** Characterized by customers with higher annual incomes and very high spending scores, regardless of age. These are prime targets for premium products and loyalty programs.
    * **Segment 2 (Frugal/Low-Value):** Consists of younger customers with lower annual incomes and lower spending scores. This segment might benefit from introductory offers or value-oriented promotions.

These insights enable the mall to tailor marketing campaigns, product offerings, and customer services more effectively for each specific group.

---

## Deployment

The final trained K-Means clustering model, along with the fitted `StandardScaler` and the list of expected feature names, has been saved using `joblib`. The model is prepared for deployment as a **Flask API**. This API can receive new customer data (in JSON format) and return a predicted cluster label, enabling real-time customer segmentation. The API is designed to handle both single and batch predictions, applying the exact same preprocessing pipeline (feature scaling and feature selection) to new input data as was used during training.

---

## How to Run the Project

1.  **Clone the Repository:** Obtain the project files from its version control repository. Navigate into the project directory.
2.  **Set up Virtual Environment (Recommended):** Create and activate a Python virtual environment to manage project dependencies in isolation.
3.  **Install Dependencies:** Install all required Python libraries listed in the project's requirements.
4.  **Data Preparation & Model Training (Jupyter Notebook):**
    * Place the customer dataset file (e.g., `Mall_Customers.csv`) in your project directory.
    * Open and run the Jupyter Notebook (`.ipynb` file) cell by cell. This notebook handles all data loading, preprocessing, clustering, and saves the necessary model artifacts (`.joblib` files).
    * Ensure the notebook executes successfully and the K-Means model, scaler, and feature list files are generated in your project root.
5.  **Run the Flask API:**
    * Ensure the `app.py` file (containing the Flask API code) is in the same directory as your saved model artifacts.
    * Open your terminal in the project directory and execute the command to start the Flask development server. The API will typically run on a local host address.
6.  **Test the API with Postman:**
    * Use a tool like Postman to send `POST` requests to the API's prediction endpoint (e.g., `/predict_cluster`).
    * The request body should be a JSON object (or a list of JSON objects for batch predictions) containing the raw input features for a new customer (e.g., `Age`, `Annual Income`, `Spending Score`).
    * The API will return a JSON response with the predicted cluster label.

---

## Technologies Used

* **Python**
* **Pandas:** Data manipulation.
* **NumPy:** Numerical operations.
* **Matplotlib & Seaborn:** Data visualization.
* **Scikit-learn:** K-Means clustering, data splitting, feature scaling.
* **Tabulate:** For enhanced table output.
* **Joblib:** Model persistence.
* **Flask:** API development.

