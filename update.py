from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

app = Flask("Humpty - Your Doctor")

# Load training data
df = pd.read_csv("Training.csv")

# Create disease-to-index mapping
disease_mapping = {disease: i for i, disease in enumerate(df['prognosis'].unique())}
df.replace({'prognosis': disease_mapping}, inplace=True)

# Create reverse mapping (index to disease)
reverse_mapping = {i: disease for disease, i in disease_mapping.items()}

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

    # Convert numeric predictions back to disease names
    dt_prediction = reverse_mapping[dt_index]
    rf_prediction = reverse_mapping[rf_index]
    nb_prediction = reverse_mapping[nb_index]

    return render_template("index.html", symptoms=symptoms, 
                           dt_prediction=dt_prediction, 
                           rf_prediction=rf_prediction, 
                           nb_prediction=nb_prediction,
                           warning="This is a prediction. Please consult a doctor for professional medical advice.")

if __name__ == '__main__':
    app.run(debug=True)
