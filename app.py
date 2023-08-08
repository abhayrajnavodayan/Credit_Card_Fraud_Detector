from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    result = None

    if request.method == 'POST':
        features = [
            float(request.form.get(f'V{i}')) for i in range(1, 30)
        ]

        model = joblib.load('credit_card_model')
        y_pred = model.predict([features])

        result = 'Normal Transaction' if y_pred == 0 else 'Fraudulent Transaction'

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
