from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open('heartdisease_model.sav', 'rb'))

@app.route('/')
def start():
    return render_template('home.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/predict_page', methods=['POST'])
def predict_page():
    try:
        # Collect input values from form
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        chest_pain_type = float(request.form['chest_pain_type'])
        resting_blood_pressure = float(request.form['resting_blood_pressure'])
        cholesterol = float(request.form['cholesterl'])
        fasting_blood_sugar = float(request.form['fasting_blood_sugar'])
        rest_ecg = float(request.form['rest_ecg'])
        max_heart_rate = float(request.form['Max_heart_rate'])
        exercise_induced_angina = float(request.form['exercise_induced_angina'])
        oldpeak = float(request.form['oldpeak'])
        slope = float(request.form['slope'])
        vessels_colored_by_flourosopy = float(request.form['vessels_colored_by_flourosopy'])
        thalassemia = float(request.form['thalassemia'])

        # Feature list (must match training data)
        feature_names = [
            'age', 'sex', 'chest_pain_type', 'resting_blood_pressure',
            'cholestoral', 'fasting_blood_sugar', 'rest_ecg',
            'Max_heart_rate', 'exercise_induced_angina',
            'oldpeak', 'slope', 'vessels_colored_by_flourosopy', 'thalassemia'
        ]

        input_data = pd.DataFrame([[age, sex, chest_pain_type, resting_blood_pressure, cholesterol,
                                    fasting_blood_sugar, rest_ecg, max_heart_rate,
                                    exercise_induced_angina, oldpeak, slope,
                                    vessels_colored_by_flourosopy, thalassemia]],
                                  columns=feature_names)

        # Predict
        prediction = model.predict(input_data)[0]

        if prediction == 0:
            result = "⚠️ Heart Disease Detected"
            color = "red"
        else:
            result = "✅ No Heart Disease Detected"
            color = "green"

        # Render a new result page
        return render_template('result.html', result=result, color=color)

    except Exception as e:
        return render_template('result.html', result=f"Error: {str(e)}", color="orange")

if __name__ == "__main__":
    app.run(debug=True)
