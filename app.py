from flask import Flask, render_template, url_for, request
from flask import session
import pandas as pd
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask_caching import Cache
from sklearn.preprocessing import StandardScaler, LabelEncoder
from front_end_utils.utils import show_heat_map, show_pie_chart, tracking_trends, prediction_pie
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.utils import load_object

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache'})
app.secret_key = "your_secret_key"
@app.route('/')
def home():
    return render_template('components/home.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        data = CustomData(
            industrial_risk=request.form.get("industrial_risk"),
            management_risk=request.form.get("management_risk"),
            financial_flexibility=request.form.get("financial_flexibility"),
            credibility=request.form.get("credibility"),
            competitiveness=request.form.get("competitiveness"),
            operating_risk=request.form.get("operating_risk")
        )
        
        prediction_df = data.get_data_as_data_frame()

        with open('Notebook/pickle_files/random_forest_classifier_model.pkl', 'rb') as file:
            model = pickle.load(file)

        res = model.predict(prediction_df)
        probability = model.predict_proba(prediction_df)
        
        prob_yes = round(probability[0][0] * 100, 2)
        prob_not = round(100 - prob_yes, 2)

        prob_pie_chart = prediction_pie(prob_yes, prob_not)

        if prob_yes < prob_not:
            final_out = f"Congratulations! There is a {prob_not}% chance of stability and only {prob_yes}% risk of bankruptcy."
        else:
            final_out = f"Unfortunately, there is a {prob_yes}% chance of bankruptcy and only {prob_not}% chance of stability."

        session['result'] = final_out
        session['prob_pie'] = prob_pie_chart  # Store pie chart in session

        return render_template('components/predict.html', page="predict", result=final_out, prob_pie=prob_pie_chart)

    # Clear session on page load
    session.pop('result', None)
    session.pop('prob_pie', None)

    return render_template('components/predict.html', page="predict", result=None, prob_pie=None)

@app.route('/analyse', methods=['POST', 'GET'])
def analyse():
    heatmap_image = show_heat_map()
    pie_chart = show_pie_chart()
    histograms = tracking_trends()
    return render_template('components/analyse.html', heatmap_image=heatmap_image, pie_chart=pie_chart, histograms=histograms)


if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)