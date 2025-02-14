from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')



if __name__=='__main__':
    app.run(debug=True)