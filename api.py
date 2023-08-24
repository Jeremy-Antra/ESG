from flask import Flask, render_template, request, session, redirect
# packages
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np
# import matplotlib.pyplot as plt


app = Flask(__name__)
app.secret_key = 'ESG'
num_of_company = 1
min_rev, max_rev = 500000, 90000000
esg_cols = 18

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

def df_to_json(df):
    return df.to_json(oriend="records", mimetype='application/json')

def make_predictions(df):
    esg, stock, stock_esg = load_models()
    columns_to_standardize = df.columns.tolist()[4:4+esg_cols]
    predicted_esg_score = esg.predict(df[columns_to_standardize])
    predicted_stock = stock.predict(df['I3: Revenue'].to_numpy().reshape(-1, 1))
    predicted_stock_esg = stock_esg.predict(df[columns_to_standardize + ['I3: Revenue']])
    pred = {
        'Predicted ESG Score' : [predicted_esg_score[0]],
        'Predicted Stock Price from Revenue': [format_stock(predicted_stock[0])],
        'Predicted Stock Price from Revenue with ESG': [format_stock(predicted_stock_esg[0])]
    }
    return pd.DataFrame(pred).round(2)

def get_input_fileds(df):
    return [[i, c] for i, c in enumerate(df.columns.to_list())] 

def get_top_bottom(df):
    columns_to_standardize = df.columns.tolist()[4:4+esg_cols]
    columns_to_rank = [col for col in df.columns if 'Difference' in col]
    kv = {col[:2]:col for col in columns_to_standardize}
    ranked_df = df[columns_to_rank].apply(lambda row: row.rank(ascending=False), axis=1)

    bottom_three = ranked_df.apply(lambda row: row.nlargest(3).index.tolist(), axis=1).to_list()[0]
    top_three = ranked_df.apply(lambda row: row.nsmallest(3).index.tolist(), axis=1).to_list()[0]

    bottom_three_col = [kv[c[:2]] for c in bottom_three]
    top_three_col = [kv[c[:2]] for c in top_three]

    t = ['Top 1', 'Top 2', 'Top 3', 'Bottom 3', 'Bottom 2', 'Bottom 1']

    ret = pd.DataFrame({k:[v] for (k, v) in zip(t, top_three_col + bottom_three_col[::-1])})

    return ret

def get_difference(df):
    # Calculate weighted ESG score for each row
    weights = {'E': 0.4, 'S': 0.3, 'G': 0.3}
    columns_to_standardize = df.columns.tolist()[4:4+esg_cols]
    for k in weights.keys():
        name = k + ' Average Score'
        lst = [col for col in columns_to_standardize if col[0] == k]
        size = len(lst)
        df[name] = df.apply(lambda row: sum(row[col] for col in columns_to_standardize if col[0] == k) / size, axis=1)
        for cur in lst:
            cur_name = cur[:2] + ' Difference than Average'
            df[cur_name] = df[cur] - df[name]
    return df

def get_suggestion(df, point):
    columns_to_standardize = df.columns.tolist()[4:4+esg_cols]
    columns_to_rank = [col for col in df.columns if 'Difference' in col]
    kv = {col[:2]:col for col in columns_to_standardize}
    ranked_df = df[columns_to_rank].apply(lambda row: row.rank(ascending=False), axis=1)

    bottom_three = ranked_df.apply(lambda row: row.nlargest(3).index.tolist(), axis=1).to_list()[0]
    bottom_three_col = [kv[c[:2]] for c in bottom_three]

    df[bottom_three_col] += point

    ret = make_predictions(df)

    return ret

@app.route('/', methods=['GET'])
def index():
    df_input = generate_input_df()
    session['df_input'] = df_input.to_json()
    return render_template('index.html', df_html=df_to_html(df_input))

@app.route('/predictData', methods=['GET'])
def generate_prediction_data():
    input = generate_input()
    # simulate input
    df_input = generate_input_df()
    # make prediction   
    esg, stock, stock_esg = load_models()
    columns_to_standardize = df_input.columns.tolist()[4:4+esg_cols]
    predicted_esg_score = esg.predict(df_input[columns_to_standardize])
    predicted_stock = stock.predict(df_input['I3: Revenue'].to_numpy().reshape(-1, 1))
    predicted_stock_esg = stock_esg.predict(df_input[columns_to_standardize + ['I3: Revenue']])
    pred = {
        'Predicted ESG Score' : [predicted_esg_score[0]],
        'Predicted Stock Price from Revenue': [format_stock(predicted_stock[0])],
        'Predicted Stock Price from Revenue with ESG': [format_stock(predicted_stock_esg[0])]
    }


    #make top bottom
    diff_df = get_difference(df_input.copy())

    columns_to_standardize = diff_df.columns.tolist()[4:4+esg_cols]
    columns_to_rank = [col for col in diff_df.columns if 'Difference' in col]
    kv = {col[:2]:col for col in columns_to_standardize}
    ranked_df = diff_df[columns_to_rank].apply(lambda row: row.rank(ascending=False), axis=1)

    bottom_three = ranked_df.apply(lambda row: row.nlargest(3).index.tolist(), axis=1).to_list()[0]
    top_three = ranked_df.apply(lambda row: row.nsmallest(3).index.tolist(), axis=1).to_list()[0]

    bottom_three_col = [kv[c[:2]] for c in bottom_three]
    top_three_col = [kv[c[:2]] for c in top_three]

    t = ['Top 1', 'Top 2', 'Top 3', 'Bottom 3', 'Bottom 2', 'Bottom 1']

    top_bottom = {k:[v] for (k, v) in zip(t, top_three_col + bottom_three_col[::-1])}

    #make suggestion
    point = random.randint(1, 10)
    suggest = get_suggestion(diff_df, point)

    columns_to_standardize = diff_df.columns.tolist()[4:4+esg_cols]
    columns_to_rank = [col for col in diff_df.columns if 'Difference' in col]
    kv = {col[:2]:col for col in columns_to_standardize}
    ranked_df = diff_df[columns_to_rank].apply(lambda row: row.rank(ascending=False), axis=1)

    bottom_three = ranked_df.apply(lambda row: row.nlargest(3).index.tolist(), axis=1).to_list()[0]
    bottom_three_col = [kv[c[:2]] for c in bottom_three]

    diff_df[bottom_three_col] += point

    esg, stock, stock_esg = load_models()
    columns_to_standardize = diff_df.columns.tolist()[4:4+esg_cols]
    predicted_esg_score = esg.predict(diff_df[columns_to_standardize])
    predicted_stock = stock.predict(diff_df['I3: Revenue'].to_numpy().reshape(-1, 1))
    predicted_stock_esg = stock_esg.predict(diff_df[columns_to_standardize + ['I3: Revenue']])
    suggest = {
        'Predicted ESG Score' : [predicted_esg_score[0]],
        'Predicted Stock Price from Revenue': [format_stock(predicted_stock[0])],
        'Predicted Stock Price from Revenue with ESG': [format_stock(predicted_stock_esg[0])]
    }
    return [input,pred,top_bottom,suggest]

@app.route('/insert', methods=['GET', 'POST'])
def insert_company(userinput):

    if request.method == 'POST':
        comp = userinput if userinput else request.form
        df_input = pd.DataFrame.from_dict(dict(comp), orient='index').transpose()
        session['df_input'] = df_input.to_json()
        return redirect('/predict')
    
    df_input = pd.read_json(session['df_input'])
    columns = get_input_fileds(df_input)
    return render_template('insert.html', input_fields = columns)



if __name__ == '__main__':
    app.run(debug=True)
