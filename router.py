import numpy as np
import pandas as pd
import joblib


from flask import Flask, render_template, request
from utils import preprocess

app = Flask(__name__)


model = joblib.load('model_XGBoost.pkl')

#Home
@app.route('/')
def home():
    return render_template('index.html')



#Predict
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        long = float(request.form['long'])
        latit = float(request.form['lat'])
        med_age = float(request.form['med_age'])
        total_rooms = float(request.form['total_rooms'])
        total_bed_rooms = float(request.form['total_bed_rooms'])
        popul = float(request.form['popul'])
        households = float(request.form['household'])
        median_income = float(request.form['median_income'])
        
        ocean_proximity = request.form['ocean_proximity']
        rooms_per_hold = total_rooms/households
        bedrooms_per_rooms = total_bed_rooms / total_rooms
        pop_per_hold = popul /households

        X_new = pd.DataFrame({'longitude':[long], 'latitude':[latit], 'housing_median_age':[med_age],'total_rooms':[total_rooms],
                              'total_bedrooms':[total_bed_rooms], 'population':[popul],
                              'households':[households],'median_income':[median_income] ,'ocean_proximity':[ocean_proximity],
                              'rooms_per_household':[rooms_per_hold],'bedrooms_per_rooms':[bedrooms_per_rooms] ,'population_per_household':[pop_per_hold]})
        
        X_processed = preprocess(X_new)
        y_pred_new = model.predict(X_processed)
        y_pred_new = '{:.4f}'.format(y_pred_new[0])

        return render_template('predict.html', pred_vals=y_pred_new)
    else:    
        return render_template('predict.html')

#about
@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)



