from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import numpy as np
from collections import Counter

app = Flask(__name__)

# Load the dataset (make sure the dataset file is in the same directory as app.py)
data = pd.read_csv('C:/Users/chint/OneDrive/Desktop/ml project(3-1)/dataset/Training.csv')
symptoms = data.columns[:-1]  # All columns except 'prognosis'
X = data[symptoms]
y = data['prognosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the models
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Create a mapping of symptoms to diseases dynamically (assuming your dataset is already loaded)
symptom_to_disease = {}

# Iterate through the dataset to build the mapping
for index, row in data.iterrows():
    disease = row['prognosis']
    for symptom, value in row[symptoms].items():
        if value == 1:  # Symptom is present
            if symptom not in symptom_to_disease:
                symptom_to_disease[symptom] = set()
            symptom_to_disease[symptom].add(disease)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', symptoms=symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    user_symptoms = request.form.getlist('symptoms')  # Get selected symptoms from form
    if not user_symptoms:
        return render_template('index.html', symptoms=symptoms, prediction="Please select at least one symptom.")

    # Convert symptoms to a binary vector
    user_symptom_vector = [1 if symptom in user_symptoms else 0 for symptom in symptoms]
    user_symptoms_df = pd.DataFrame([user_symptom_vector], columns=symptoms)

    # Create a set to hold the diseases associated with the selected symptoms
    selected_diseases = Counter()

    # Iterate through selected symptoms and count how many times each disease is linked to them
    for symptom in user_symptoms:
        if symptom in symptom_to_disease:
            for disease in symptom_to_disease[symptom]:
                selected_diseases[disease] += 1

    # Check if the symptoms belong to only one disease category
    if len(selected_diseases) == 1:
        disease = selected_diseases.most_common(1)[0][0]  # Get the disease with the highest count
        # Proceed with the prediction logic (using the model)
        return render_template('index.html', symptoms=symptoms, prediction=f"The predicted disease is: {disease}")
    
    # If multiple diseases are selected, check if there's a clear dominant disease
    if len(selected_diseases) > 1:
        # Check if the selected symptoms are overwhelmingly related to one disease
        dominant_disease = selected_diseases.most_common(1)[0][0]
        # If the most common disease has a higher count than a certain threshold, it's a valid prediction
        if selected_diseases[dominant_disease] > len(user_symptoms) / 2:
            return render_template('index.html', symptoms=symptoms, prediction=f"The predicted disease is: {dominant_disease}")
        else:
            return render_template('index.html', symptoms=symptoms, prediction="Please select symptoms from only one disease category at a time.")

    # Default message if no valid disease category is selected
    return render_template('index.html', symptoms=symptoms, prediction="Please select symptoms from only one disease category at a time.")
