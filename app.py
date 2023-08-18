from flask import Flask, render_template, request, session, redirect
# packages
import pandas as pd
import os
import psycopg2
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import joblib
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt


app = Flask(__name__)
app.secret_key = 'ESG'
num_of_company = 1
min_rev, max_rev = 500000, 90000000

def generate_input_df():
    data = {
    'I1: Company Name': ['Company Example'],
    'I2: Number of Employee': [random.randint(50, 2000) for _ in range(num_of_company)],
    'I3: Revenue': [random.randint(min_rev, max_rev) for _ in range(num_of_company)],
    'I4: Year' : [2023],
    'E1: Carbon emissions and intensity': [random.randint(1, 100) for _ in range(num_of_company)],
    'E2: Energy consumption and renewable energy use': [random.randint(1, 100) for _ in range(num_of_company)],
    'E3: Water usage and management': [random.randint(1, 100) for _ in range(num_of_company)],
    'E4: Waste generation and recycling rates': [random.randint(1, 100) for _ in range(num_of_company)],
    'E5: Air and water pollution levels': [random.randint(1, 100) for _ in range(num_of_company)],
    'E6: Biodiversity conservation efforts': [random.randint(1, 100) for _ in range(num_of_company)],
    'S1: Employee diversity and inclusion': [random.randint(1, 100) for _ in range(num_of_company)],
    'S2: Labor practices and working conditions': [random.randint(1, 100) for _ in range(num_of_company)],
    'S3: Health and safety performance': [random.randint(1, 100) for _ in range(num_of_company)],
    'S4: Employee turnover and satisfaction': [random.randint(1, 100) for _ in range(num_of_company)],
    'S5: Community engagement and philanthropic activities': [random.randint(1, 100) for _ in range(num_of_company)],
    'S6: Product quality and safety': [random.randint(1, 100) for _ in range(num_of_company)],
    'G1: Board diversity and composition': [random.randint(1, 100) for _ in range(num_of_company)],
    'G2: Executive compensation and incentives': [random.randint(1, 100) for _ in range(num_of_company)],
    'G3: Transparency of executive pay ratios': [random.randint(1, 100) for _ in range(num_of_company)],
    'G4: Shareholder rights and engagement': [random.randint(1, 100) for _ in range(num_of_company)],
    'G5: Ethical business conduct and anti-corruption policies': [random.randint(1, 100) for _ in range(num_of_company)],
    'G6: Risk management and regulatory compliance': [random.randint(1, 100) for _ in range(num_of_company)]
    }

    return pd.DataFrame(data)

def load_models():
    esg = joblib.load('assets/linear_ESG_model.pkl')
    stock = joblib.load('assets/linear_Stock_model.pkl')
    stock_esg = joblib.load('assets/linear_Stock_ESG_model.pkl')
    return esg, stock, stock_esg

def format_stock(arr):
    headers = ['Average', 'Minimum', 'Maximum']
    return [h + ': $' + str(s) for h, s in zip (headers, np.round(arr, 2))]

def df_to_html(df):
    return df.to_html(classes='table table-striped', index=False)

def make_predictions(df):
    esg, stock, stock_esg = load_models()
    esg_cols = df.columns.tolist()[4:]
    predicted_esg_score = esg.predict(df[esg_cols])
    predicted_stock = stock.predict(df['I3: Revenue'].to_numpy().reshape(-1, 1))
    predicted_stock_esg = stock_esg.predict(df[esg_cols + ['I3: Revenue']])
    pred = {
        'Predicted ESG Score' : [predicted_esg_score[0]],
        'Predicted Stock Price from Revenue': [format_stock(predicted_stock[0])],
        'Predicted Stock Price from Revenue with ESG': [format_stock(predicted_stock_esg[0])]
    }
    return pd.DataFrame(pred).round(2)

def get_input_fileds(df):
    return [[i, c] for i, c in enumerate(df.columns.to_list())] 



@app.route('/', methods=['GET'])
def index():
    df_input = generate_input_df()
    session['df_input'] = df_input.to_json()
    return render_template('index.html', df_html=df_to_html(df_input))

@app.route('/predict', methods=['GET'])
def generate_prediction():
    df_input = pd.read_json(session['df_input'])
    pred = make_predictions(df_input)

    return render_template('predict.html', df_html=df_to_html(df_input), 
                           pred_html=df_to_html(pred))

@app.route('/insert', methods=['GET', 'POST'])
def insert_company():

    if request.method == 'POST':
        comp = request.form
        df_input = pd.DataFrame.from_dict(dict(comp), orient='index').transpose()
        print(df_input)
        session['df_input'] = df_input.to_json()
        return redirect('/predict')
    
    df_input = pd.read_json(session['df_input'])
    columns = get_input_fileds(df_input)
    return render_template('insert.html', input_fields = columns)



if __name__ == '__main__':
    app.run(debug=True)
