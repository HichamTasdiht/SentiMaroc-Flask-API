from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import re
from scipy import sparse

# Create a Flask web application instance.
app = Flask(__name__)
# This line reads the SECRET_KEY from the environment variables set on Railway
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')


# Define a route for the home page ("/")
@app.route("/")
def home():
    return render_template("index.html")

try:
    an_model = joblib.load('An_Model.joblib')
    # Load the list of categorical columns from training for the complex model.
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    cat_columns = joblib.load('cat_columns.pkl')
    print("Complex analysis model and preprocessing objects loaded successfully.")
    complex_components_loaded = True
except FileNotFoundError as e:
    print(f"Error: A file was not found: {e}")
    complex_components_loaded = False
except Exception as e:
    print(f"An error occurred while loading the complex components: {e}")
    complex_components_loaded = False

try:
    pr_model = joblib.load('Pr_Model.joblib')
    print("Simple sentiment model loaded successfully.")
    simple_model_loaded = True
except FileNotFoundError:
    print("Error: The simple sentiment model file 'another_model.joblib' was not found.")
    simple_model_loaded = False
except Exception as e:
    print(f"An error occurred while loading the simple sentiment model: {e}")
    simple_model_loaded = False


# --- Prediction Function for Complex Model (Model 1) ---
def preprocess_and_predict_with_complex_model(new_data):
    
    new_data_df = pd.DataFrame([new_data])

    if "product_name" in new_data_df.columns:
        new_data_df = new_data_df.drop(["product_name"], axis=1)

    new_data_df["date"] = pd.to_datetime(new_data_df["date"], errors="coerce")

    new_data_df["year"] = new_data_df["date"].dt.year
    new_data_df["month"] = new_data_df["date"].dt.month
    new_data_df["day"] = new_data_df["date"].dt.day
    new_data_df["dayofweek"] = new_data_df["date"].dt.dayofweek

    new_data_df = new_data_df.drop("date", axis=1)

    review_clean = re.sub(r'[^\w\s]', '', new_data_df['review_text'].iloc[0])
    
    X_text = tfidf.transform([review_clean])
    
    numeric_features = ['price_dh', 'year', 'rating']
    X_numeric = sparse.csr_matrix(new_data_df[numeric_features].values)

    categorical_features = ['brand_name', 'category', 'language', 'month', 'day', 'dayofweek']
    X_cat_df = pd.DataFrame([{f: new_data_df[f].iloc[0] for f in categorical_features}])
    X_cat = pd.get_dummies(X_cat_df)
    

    # Align categorical columns with training
    for col in cat_columns:
        if col not in X_cat.columns:
            X_cat[col] = 0
    X_cat = X_cat[cat_columns]
    X_cat_sparse = sparse.csr_matrix(X_cat.values.astype(int))

    # Combine all features
    X_final = sparse.hstack([X_numeric, X_cat_sparse, X_text])

    # Predict with the complex model
    y_pred_encoded = an_model.predict(X_final)
    print(y_pred_encoded)
    
    y_pred = "Neutre"
    if y_pred_encoded == 0 :
        y_pred = "Négatif"
    if y_pred_encoded == 2 :
        y_pred = "Positif"   
    return y_pred


# --- Prediction Function for Simple Sentiment Model (Model 2) ---
def predict_simple_sentiment(product_name, brand_name):
    combined_text = f"{product_name} {brand_name}"
    predicted_sentiment = pr_model.predict([combined_text])
    return predicted_sentiment[0]


# --- API Endpoint for Simple Sentiment Prediction ---
@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment_api():
    
    if not simple_model_loaded:
        return jsonify({'error': 'Simple sentiment model not available.'}), 500

    data = request.get_json()
    required_keys = ['product_name', 'brand_name']
    if not data or not all(key in data for key in required_keys):
        return jsonify({'error': f'Invalid JSON format. Requires keys: {required_keys}.'}), 400

    try:
        sentiment = predict_simple_sentiment(data['product_name'], data['brand_name'])
        return jsonify({
            'product_name': data['product_name'],
            'brand_name': data['brand_name'],
            'predicted_sentiment': sentiment
        })
    except Exception as e:
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

# --- API Endpoint for Complex Analysis ---
@app.route('/analyze_review', methods=['POST'])
def analyze_text_api():

    if not complex_components_loaded:
        return jsonify({'error': 'Complex model and components not available.'}), 500

    data = request.get_json()
    data = data["data"]
    required_keys = ['product_name', 'date', 'review_text', 'price_dh', 'rating', 'brand_name', 'category', 'language']
    if not data or not all(key in data for key in required_keys):
        return jsonify({'error': f'Invalid JSON format. Requires keys: {required_keys}.'}), 400

    try:
        analysis_result = preprocess_and_predict_with_complex_model(data)
        return jsonify({
            'input_data': data,
            'analysis_result': analysis_result
        })
    except Exception as e:
        return jsonify({'error': f'An error occurred during prediction : {str(e)}'}), 500

# --- Entry Point for the Application ---
if __name__ == '__main__':
    app.run(debug=True)
