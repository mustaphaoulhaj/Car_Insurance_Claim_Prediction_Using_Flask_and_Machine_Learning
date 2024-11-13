from flask import Flask, render_template, request
import joblib  # Utiliser joblib pour charger le modèle
import numpy as np
import pandas as pd

# Initialiser l'application Flask
app = Flask(__name__)
model = joblib.load('Prediction_Model.pkl')  # Charger le modèle ici

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Début de la fonction predict")  # Debug point 1

        # Récupérer les données du formulaire
        income = float(request.form['income'])
        customer_state = request.form['customer_state']
        total_claims = float(request.form['total_claims'])
        total_claims_amount = float(request.form['total_claims_amount'])
        customer_lifetime_value = float(request.form['customer_lifetime_value'])
        highest_education = request.form['highest_education']
        employment_status = request.form['employment_status']
        gender = request.form['gender']
        residence_type = request.form['residence_type']
        marital_status = request.form['marital_status']
        sales_channel = request.form['sales_channel']
        coverage = request.form['coverage']
        vehicle_class = request.form['vehicle_class']
        vehicle_size = request.form['vehicle_size']
        
        # Affiche les données récupérées
        print("Données du formulaire:", income, customer_state, total_claims, total_claims_amount, customer_lifetime_value, highest_education, employment_status, gender, residence_type, marital_status, sales_channel, coverage, vehicle_class, vehicle_size)

        # Créer un DataFrame avec les noms de colonnes
        input_data = pd.DataFrame({
            'income': [income],
            'customer_state' : [customer_state],
            'total_claims' : [total_claims],
            'total_claims_amount' : [total_claims_amount],
            'customer_lifetime_value': [customer_lifetime_value],
            'highest_education': [highest_education],
            'employment_status': [employment_status],
            'gender': [gender],
            'residence_type': [residence_type],
            'marital_status': [marital_status],
            'sales_channel': [sales_channel],
            'coverage': [coverage],
            'vehicle_class': [vehicle_class],
            'vehicle_size': [vehicle_size]
        })

        # Vérifie si le DataFrame est bien formé
        print("DataFrame d'entrée :", input_data)

        # Faire la prédiction
        prediction = model.predict(input_data)

        # Affiche la prédiction pour vérifier
        print("Prédiction obtenue:", prediction[0])

        # Afficher le résultat
        return render_template('index.html', predicted_amount=prediction[0])

    except Exception as e:
        # Affiche l'erreur dans le terminal et dans la page
        print("Erreur lors de la prédiction:", str(e))
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)