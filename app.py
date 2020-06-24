import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import joblib as jl
app = Flask(__name__, template_folder='templates', static_folder='static')
with open(f'model/XGB Loan.pkl', 'rb') as f:
    model = jl.load(f)

@app.route('/', methods = ['GET', 'POST'])
def main():
    if request.method == 'GET':
        return (render_template('index2.html'))
    if request.method == 'POST':
        fico = request.form['fico']
        int_rate = request.form['int_rate']
        loan_amnt = request.form['loan_amnt']
        grade = request.form['grade']
        dti = request.form['dti']
        revol_util = request.form['revol_util']
        mo_sin_old_rev_tl_op = request.form['mo_sin_old_rev_tl_op']
        tot_cur_bal = request.form['tot_cur_bal']
        all_util = request.form['all_util']
        annual_inc = request.form['annual_inc']
        mths_since_recent_bc = request.form['mths_since_recent_bc']
        num_actv_bc_tl = request.form['num_actv_bc_tl']
        feature_tbl = pd.DataFrame([[fico, int_rate, loan_amnt, grade, dti, revol_util,
                                      mo_sin_old_rev_tl_op, tot_cur_bal, all_util, annual_inc,mths_since_recent_bc,
                                       num_actv_bc_tl]], columns = ['last_fico_range_avg', 'int_rate',
                                                                    'loan_amnt', 'grade', 'dti', 'revol_util',
                                                                    'mo_sin_old_rev_tl_op', 'tot_cur_bal',
                                                                    'all_util','annual_inc', 'mths_since_recent_bc',
                                                                    'num_actv_bc_tl'], dtype = float)
        prediction = model.predict(feature_tbl)[0]
        if prediction == 0:
            return render_template('index2.html', original_input={'fico':fico,'int_rate':int_rate, 'loan_amnt': loan_amnt,
                                                                        'grade': grade, 'dti': dti, 'revol_util': revol_util,
                                                                        'mo_sin_old_rev_tl_op': mo_sin_old_rev_tl_op,
                                                                        'tot_cur_bal': tot_cur_bal, 'all_util': all_util,
                                                                        'annual_inc': annual_inc, 'mths_since_recent_bc': mths_since_recent_bc,
                                                                        'num_actv_bc_tl': num_actv_bc_tl}, prediction_text="Good News! You are eligible for this loan!")
        else:
            return render_template('index2.html', original_input={'fico':fico,'int_rate':int_rate, 'loan_amnt': loan_amnt,
                                                                        'grade': grade, 'dti': dti, 'revol_util': revol_util,
                                                                        'mo_sin_old_rev_tl_op': mo_sin_old_rev_tl_op,
                                                                        'tot_cur_bal': tot_cur_bal, 'all_util': all_util,
                                                                        'annual_inc': annual_inc, 'mths_since_recent_bc': mths_since_recent_bc,
                                                                        'num_actv_bc_tl': num_actv_bc_tl}, prediction_text="We are sorry. You are not eligible for this loan.")
#@app.route('/predict', methods = ['POST', 'GET'])
#def predict():
    #data = [float(x) for x in request.form.values()]
    #feature_tbl = pd.DataFrame({"last_fico_range_avg": [data[0]],
                                 #"int_rate": [data[1]],
                                 #"loan_amnt": [data[2]],
                                 #"grade": [data[3]],
                                 #"dti": [data[4]],
                                 #"revol_util": [data[5]],
                                 #"mo_sin_old_rev_tl_op": [data[6]],
                                 #"tot_cur_bal": [data[7]],
                                 #"all_util": [data[8]],
                                 #"annual_inc": [data[9]],
                                 #"mths_since_recent_bc": [data[10]],
                                 #"num_actv_bc_tl": [data[11]]})
    #prediction = model.predict(feature_tbl)
    #output = prediction[0]
    #if output == 1:
        #return render_template('index2.html', prediction_text = "Sorry. You are not eligible for a loan.")
    #else:
        #return render_template('index2.html', prediction_text = "Great! You are eligible for a loan.")

#@app.route('/results', methods=['POST'])
#def results():
    #data = request.get_json(force = True)
    #feature = [np.array(list(data.values()))]
    #feature_tbl = pd.DataFrame({"last_fico_range_avg": feature[0],
                                 #"int_rate": feature[1],
                                 #"loan_amnt": feature[2],
                                # "grade": feature[3],
                                 #"dti": feature[4],
                                 #"revol_util": feature[5],
                                 #"mo_sin_old_rev_tl_op": feature[6],
                                 #"tot_cur_bal": feature[7],
                                 #"all_util": feature[8],
                                 #"annual_inc": feature[9],
                                 #"mths_since_recent_bc": feature[10],
                                 #"num_actv_bc_tl": feature[11]})
    #prediction = model.predict(feature_tbl)
    #output = prediction[0]
    #return jsonify(output)

if __name__ == "__main__":
    app.run(debug = True)