from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# --- Load Model, Scaler, and Feature List ---
try:
    kmeans_model = joblib.load('kmeans_customer_segmentation_model.joblib')
    scaler_clustering = joblib.load('scaler_clustering.joblib')
    clustering_features = joblib.load('clustering_features.joblib')
    print("K-Means Model, Scaler, and Features loaded successfully.")
except Exception as e:
    print(f"Error loading assets: {e}")
    # In a production environment, you might want to log this and exit or raise an error.
    exit()

# --- Preprocessing Function for New Data ---
def preprocess_for_clustering(input_data_dict: dict, trained_features: list, scaler_obj):
    """
    Preprocesses new input data for K-Means prediction.
    Ensures features are in the correct order and scaled.
    """
    # Convert input dictionary to a DataFrame, ensuring it's a single row
    input_df = pd.DataFrame([input_data_dict])

    # Reindex the input DataFrame to match the order and presence of features
    # used during training. Fill missing values with 0 if a feature is absent
    # in the input JSON, assuming 0 is a reasonable default for numerical features.
    processed_input = input_df.reindex(columns=trained_features, fill_value=0)

    # Ensure no infinite values remain, replace with NaN then fill with 0
    processed_input.replace([np.inf, -np.inf], np.nan, inplace=True)
    processed_input.fillna(0, inplace=True) # Fill any NaNs after reindexing

    # Scale the features using the *saved* scaler
    # The scaler expects a DataFrame with the same columns it was fitted on.
    scaled_input = scaler_obj.transform(processed_input)

    return scaled_input

# --- Flask API Endpoint for Prediction ---
@app.route('/predict_cluster', methods=['POST'])
def predict_cluster():
    try:
        # Get JSON data from the request body
        data = request.get_json(force=True)

        # Preprocess the input data
        # Ensure the keys in the incoming JSON match the feature names used for training
        # (e.g., "Age", "Annual Income", "Spending Score")
        processed_scaled_data = preprocess_for_clustering(data, clustering_features, scaler_clustering)

        # Make prediction (get the cluster label)
        cluster_label = kmeans_model.predict(processed_scaled_data)[0]

        # Return the predicted cluster label as JSON
        return jsonify({'predicted_cluster': int(cluster_label)}) # Convert to int for JSON serialization

    except Exception as e:
        app.logger.error(f"Error during cluster prediction: {e}", exc_info=True)
        return jsonify({'error': f"Cluster prediction failed. Please check your input data. Details: {str(e)}"}), 400

# --- Run the Flask App ---
if __name__ == '__main__':
    # To run this Flask app:
    # 1. Save this code as `app.py` in the same directory as your .joblib files.
    # 2. Open your terminal, navigate to that directory.
    # 3. Run: `flask run`
    #    For development, you can use `python app.py` (it will use Werkzeug dev server)
    app.run(debug=True, host='0.0.0.0', port=5000) # debug=True is for development only