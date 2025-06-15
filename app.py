from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Initialize Flask app
app = Flask("Humpty - Your Doctor")

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load training data
df = pd.read_csv("Training.csv")

disease_mapping = {disease: i for i, disease in enumerate(df['prognosis'].unique())}
df.replace({'prognosis': disease_mapping}, inplace=True)
reverse_mapping = {v: k for k, v in disease_mapping.items()}

X = df.iloc[:, :-1]
y = df['prognosis']

# Load testing data
tr = pd.read_csv("Testing.csv")
tr.replace({'prognosis': disease_mapping}, inplace=True)
X_test = tr.iloc[:, :-1]
y_test = tr['prognosis']

# Train models
dt_model = tree.DecisionTreeClassifier().fit(X, y)
rf_model = RandomForestClassifier().fit(X, y)
nb_model = GaussianNB().fit(X, y)

# List of symptoms
symptoms = list(X.columns)

# Precautions and specialists dictionary (Updated for full coverage)
precautions = {
    "Fungal infection": {
        "precautions": ["Keep the affected area clean and dry.", "Use antifungal creams.", "Avoid sharing personal items."],
        "specialist": "Dermatologist"
    },
    "Diabetes": {
        "precautions": ["Maintain a balanced diet.", "Exercise regularly.", "Monitor blood sugar levels."],
        "specialist": "Endocrinologist"
    },
    "Malaria": {
        "precautions": ["Use mosquito nets.", "Wear protective clothing.", "Take antimalarial medications if prescribed."],
        "specialist": "Infectious Disease Specialist"
    },
    "Dengue": {
        "precautions": ["Avoid mosquito bites.", "Stay hydrated.", "Seek medical attention if symptoms worsen."],
        "specialist": "General Physician"
    },
    "Common Cold": {
        "precautions": ["Drink warm fluids.", "Rest well.", "Avoid cold weather."],
        "specialist": "General Physician"
    },
    "Bronchial Asthma": {
        "precautions": ["Avoid allergens.", "Use prescribed inhalers.", "Maintain a clean environment."],
        "specialist": "Pulmonologist"
    },
    "Typhoid": {
        "precautions": ["Drink clean water.", "Maintain hygiene.", "Eat well-cooked food."],
        "specialist": "General Physician"
    },
    "GERD": {
        "precautions": ["Eat smaller meals.", "Avoid spicy and acidic foods.", "Don't lie down immediately after eating."],
        "specialist": "Gastroenterologist"
    }
}

def get_gemini_response(disease):
    """Get AI-generated insights for a disease using Gemini AI."""
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(f"Provide detailed medical insights for {disease}. Include symptoms, causes, treatments, and precautions.")
    return response.text if response else "No AI response available."

@app.route('/')
def home():
    return render_template("index.html", symptoms=symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    selected_symptoms = request.form.getlist('symptoms')
    input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptoms]

    dt_index = int(dt_model.predict([input_vector])[0])
    rf_index = int(rf_model.predict([input_vector])[0])
    nb_index = int(nb_model.predict([input_vector])[0])

    dt_prediction = reverse_mapping.get(dt_index, "Unknown Disease").strip()
    rf_prediction = reverse_mapping.get(rf_index, "Unknown Disease").strip()
    nb_prediction = reverse_mapping.get(nb_index, "Unknown Disease").strip()

    dt_precaution = precautions.get(dt_prediction, {"precautions": ["No specific precautions found."], "specialist": "Consult a doctor"})
    ai_response = get_gemini_response(dt_prediction)
    
    return render_template("index.html", symptoms=symptoms, 
                           dt_prediction=dt_prediction, 
                           rf_prediction=rf_prediction, 
                           nb_prediction=nb_prediction,
                           precautions=dt_precaution["precautions"],
                           specialist=dt_precaution["specialist"],
                           ai_response=ai_response,
                           warning="⚠ This is a prediction. Please consult a doctor for professional medical advice.")

if __name__ == '__main__':
    app.run(debug=True)
