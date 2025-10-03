# =========================
# Flask AI Disease Prediction App
# =========================

# Importing required libraries
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
import logging
import time

# =========================
# Initialize Flask app
# =========================
app = Flask(__name__)

# =========================
# Setup Logging
# =========================
# All requests and predictions will be logged in app.log
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# =========================
# Load Datasets
# =========================
sym_des = pd.read_csv("datasets/symptoms_df.csv")   # Symptom to disease mapping
precautions = pd.read_csv("datasets/precautions_df.csv")  # Disease-specific precautions
workout = pd.read_csv("datasets/workout_df.csv")    # Disease-specific workout recommendations
description = pd.read_csv("datasets/description.csv")  # Disease descriptions
medications = pd.read_csv('datasets/medications.csv')  # Disease-specific medications
diets = pd.read_csv("datasets/diets.csv")  # Recommended diets for diseases

# =========================
# Load Pre-trained Model
# =========================
svc = pickle.load(open('svc.pkl', 'rb'))

# =========================
# Helper Functions
# =========================

def get_description(disease):
    """Return the description of the disease"""
    desc_list = description[description['Disease']==disease]['Description'].values
    if len(desc_list) == 0:
        return "No description available for this disease."
    return " ".join(desc_list)

def get_precautions(disease):
    """Return a list of precautions for the disease"""
    pre = precautions[precautions['Disease']==disease][['Precaution_1','Precaution_2','Precaution_3','Precaution_4']].values.tolist()
    if not pre:
        return ["No precautions available."]
    return pre[0]

def get_medications(disease):
    """Return a list of medications for the disease"""
    meds = medications[medications['Disease']==disease]['Medication'].tolist()
    return meds if meds else ["No medications available."]

def get_diet(disease):
    """Return recommended diets for the disease"""
    die = diets[diets['Disease']==disease]['Diet'].tolist()
    return die if die else ["No diet recommendation available."]

def get_workout(disease):
    """Return recommended workouts for the disease"""
    wrk = workout[workout['disease']==disease]['workout'].tolist()
    return wrk if wrk else ["No workout recommendation available."]

def helper(dis):
    """
    Combines all information about a predicted disease:
    Description, Precautions, Medications, Diet, Workout
    """
    return get_description(dis), get_precautions(dis), get_medications(dis), get_diet(dis), get_workout(dis)

def clean_symptoms(symptoms):
    """
    Clean and normalize user input symptoms:
    - Convert to lowercase
    - Remove extra spaces
    - Replace spaces with underscores
    """
    return [s.strip().lower().replace(" ", "_") for s in symptoms.split(',')]

def vectorize_symptoms(symptoms):
    """
    Convert list of symptom strings into a binary input vector for the model
    """
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in symptoms:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
    return input_vector

def predict_disease(symptoms):
    """
    Predict the disease based on user input symptoms
    """
    input_vector = vectorize_symptoms(symptoms)
    predicted_index = svc.predict([input_vector])[0]
    disease_name = diseases_list.get(predicted_index, "Unknown Disease")
    return disease_name

# =========================
# Symptoms and Diseases Dictionaries
# =========================
# (Same as original; omitted here for brevity but should be included in full code)
symptoms_dict = { ... }  # keep your full symptom mapping here
diseases_list = { ... }  # keep your full disease mapping here

# =========================
# Timing Decorator for routes
# =========================
def timeit(func):
    """Decorator to time how long a request takes"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(f"Function {func.__name__} took {end-start:.4f} seconds")
        return result
    return wrapper

# =========================
# Flask Routes
# =========================

@app.route('/')
@timeit
def index():
    """Render the home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@timeit
def predict():
    """Handle prediction request from user"""
    symptoms = request.form.get('symptoms')
    logging.info(f"Received symptoms: {symptoms}")

    if not symptoms or symptoms.lower() == 'symptoms':
        return render_template('index.html', message="Please enter valid symptoms.")

    # Clean and normalize user input
    user_symptoms = clean_symptoms(symptoms)
    
    # Get predicted disease
    predicted_disease = predict_disease(user_symptoms)
    logging.info(f"Predicted disease: {predicted_disease}")

    # Get all related info
    dis_des, my_precautions, medications_list, rec_diet, wrkout = helper(predicted_disease)

    # Render results back to home page
    return render_template('index.html', predicted_disease=predicted_disease, dis_des=dis_des,
                           my_precautions=my_precautions, medications=medications_list,
                           my_diet=rec_diet, workout=wrkout)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

# =========================
# Run Flask App
# =========================
if __name__ == '__main__':
    app.run(debug=True)
