from flask import Flask, render_template, request
import re
import pandas as pd
import numpy as np
import copy
import joblib
import psycopg2
from sqlalchemy import create_engine
  
# creating Engine which connect to postgreSQL
conn_string = 'postgresql+psycopg2://postgres:postgres@localhost:5432/sms_db'
  
db = create_engine(conn_string)
conn = db.connect()

# Load the saved model
model = joblib.load('processed1')

# Define flask
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST' :
        f = request.files['file']
        email_data = pd.read_excel(f)
               
        test_pred_lap = pd.DataFrame(model.predict(email_data.text))
        test_pred_lap.columns = ["spam_pred"]

        final = pd.concat([email_data, test_pred_lap], axis = 1)
        

        final.to_sql('sms_predictions', con = conn, if_exists = 'replace', index= False)
        conn.autocommit = True
               
        return render_template("new.html", Y = final.to_html(justify = 'center'))

if __name__=='__main__':
    app.run(debug = True)
