from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the pickled SVM model
with open('pm_model-1.pkl', 'rb') as model_file:
    svm_model = pickle.load(model_file)
    
with open('vector-1.pickle', 'rb') as f:
    # Load the pickle object
    vectorizer = pickle.load(f)


# Endpoint to receive POST requests with ingredients and return predictions
@app.route('/', methods=['POST'])
def predict1():
    try:
        data = request.get_json()
        
        new_ingredients_vector = vectorizer.transform(data)
        

        # Make predictions
        predictions = svm_model.predict(new_ingredients_vector)

        # Prepare and return the predictions as JSON
        return jsonify( predictions.tolist())
    except Exception as e:
        return jsonify({'error': str(e)})

