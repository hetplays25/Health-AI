from flask import Flask, render_template, request
import pandas as pd
import pickle

# ------------------------------
# INIT APP
# ------------------------------
app = Flask(__name__)

# ------------------------------
# LOAD MODEL + ENCODER
# ------------------------------
model = pickle.load(open("model.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

# Feature columns (must match dataset EXACTLY)
columns = [
    'fever','cough','fatigue','headache','sore_throat',
    'runny_nose','sneezing','body_pain','nausea',
    'chills','breathlessness','temperature','heart_rate'
]

# ------------------------------
# HOME ROUTE
# ------------------------------
@app.route('/')
def home():
    return render_template("index.html")

# ------------------------------
# PREDICT ROUTE
# ------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get inputs
        symptoms_text = request.form.get("symptoms", "").lower()
        user_symptoms = symptoms_text.split()

        temperature = float(request.form.get("temperature", 98))
        heart_rate = float(request.form.get("heart_rate", 72))

        # Create input vector
        input_vector = []

        for col in columns:
            if col == "temperature":
                input_vector.append(temperature)
            elif col == "heart_rate":
                input_vector.append(heart_rate)
            elif col in user_symptoms:
                input_vector.append(1)
            else:
                input_vector.append(0)

        # Convert to DataFrame
        input_df = pd.DataFrame([input_vector], columns=columns)

        # Predict probabilities
        probs = model.predict_proba(input_df)[0]

        # Get top 2 predictions
        top_indices = probs.argsort()[-2:][::-1]

        results = []
        for i in top_indices:
            disease = le.inverse_transform([i])[0]
            confidence = round(probs[i] * 100, 2)
            results.append(f"{disease} ({confidence}%)")

        # ------------------------------
        # SEVERITY LOGIC
        # ------------------------------
        symptom_count = sum([1 for val in input_vector[:-2] if val == 1])

        if symptom_count <= 2:
            severity = "Mild"
        elif symptom_count <= 5:
            severity = "Moderate"
        else:
            severity = "Severe"

        # ------------------------------
        # RECOMMENDATION
        # ------------------------------
        if severity == "Severe" or temperature > 102 or heart_rate > 110:
            advice = "Consult a doctor immediately"
        elif severity == "Moderate":
            advice = "Take rest, hydration, and monitor symptoms"
        else:
            advice = "No major issue, stay healthy"

        # Return result
        return render_template(
            "index.html",
            prediction=results,
            severity=severity,
            advice=advice
        )

    except Exception as e:
        return render_template("index.html", error=str(e))


# ------------------------------
# RUN APP
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)