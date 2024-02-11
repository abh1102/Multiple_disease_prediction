import pickle

import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
app = Flask(__name__)


#input_data=(38,1,2,138,175,0,1,173,0,0,2,4,2)
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/cancer')
def cancer ():
    return render_template('cancer.html')
@app.route("/predict_cancer", methods=['POST'])
def predict_cancer():  
    model=pickle.load(open('Model/model_cancer.pkl','rb'))
    scaler = joblib.load('Model/model_scaler.pkl')
    Radius_mean=float(request.form['radius Mean'])
    Texture_mean=float(request.form['texture Mean'])
    Perimeter_mean=float(request.form['perimeter Mean'])
    Area_mean=float(request.form['area Mean'])
    Smoothness_mean=float(request.form['smoothness Mean'])
    Compactness_mean=float(request.form['compactness Mean'])
    Concavity_mean=float(request.form['concavity Mean'])
    Concave_Points_mean=float(request.form['concave points Mean'])
    Symmetric_Mean=float(request.form['symmetric Mean'])
    fractal_dimension_Mean=float(request.form['fractal Mean'])
    Radius_se=float(request.form['radius'])
    Texture_se=float(request.form['texture'])
    perimeter_se=float(request.form['perimeter'])
    Area_se=float(request.form['area se'])
    Smoothness_se=float(request.form['smoothness'])
    Compactness_se=float(request.form['compactness'])
    Concavity_se=float(request.form['concavity'])
    Concave_point_se=float(request.form['Concave points_se'])
    Symmetry=float(request.form['symmetry_se'])
    Fractal_dimension_se=float(request.form['fractal_dimension_se'])
    Raddius_worst=float(request.form['radius_worst'])
    Texture_worst=float(request.form['texture_worst'])
    Perimeter_Worst=float(request.form['perimeter_worst'])
    Area_Worst=float(request.form['area_worst'])
    smoothness_Worst=float(request.form['smoothness_worst'])
    Compactness_Worst=float(request.form['compactness_worst'])
    Concavity_Worst=float(request.form['concavity_worst'])
    Concave_Points_Worst=float(request.form['concave points_worst'])
    Symmetry_Worst=float(request.form['Symmetry_worst'])
    Fractal_Dimension_Worst=float(request.form['Fractal_dimension_worst'])
    
    int_features= np.array([Radius_mean,  Texture_mean, Perimeter_mean,Area_mean,  Smoothness_mean, Compactness_mean,Concavity_mean,Concave_Points_mean, Symmetric_Mean,fractal_dimension_Mean,Radius_se,Texture_se
        ,perimeter_se,Area_se,Smoothness_se, Compactness_se,Concavity_se,Concave_point_se, Symmetry,Fractal_dimension_se, Raddius_worst, Texture_worst,Perimeter_Worst, Area_Worst,smoothness_Worst,Compactness_Worst,
    Concavity_Worst, Concave_Points_Worst,Symmetry_Worst, Fractal_Dimension_Worst
        ])
    features = [np.array(int_features)]
    final_features = scaler.transform(features)
    prediction = model.predict(final_features)
    output = prediction
    print(prediction)
    prediction = model.predict(final_features)
    if(output==0):
         return render_template('cancer.html',prediction="you are not suffering from cancer{}".format(output))
    else:
       # prediction="you are  not suffering from heart disease"
        return render_template('cancer.html',prediction="you are suffering from cancer{}".format(output))
   
@app.route('/diabetes')
def diabetes():
    return render_template('diabetse.html')
@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes(): 
    model=pickle.load(open('Model/model.pkl','rb'))
    scaler = joblib.load('Model/model_scaler1.pkl')

    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    features = [np.array(int_features)]
    final_features = scaler.transform(features)
    prediction = model.predict(final_features)

    output = prediction
    
    if output == 1:
        prediction='You Have Diabetes' 
        return render_template('diabetse.html', prediction=prediction, preg=int_features[0], gluc=int_features[1], bp=int_features[2], st=int_features[3], insu=int_features[4], bmi=int_features[5], dpf=int_features[6], age=int_features[7])
    else:
        prediction='You Do Not Have Diabetes'
        return render_template('diabetse.html', prediction=prediction, preg=int_features[0], gluc=int_features[1], bp=int_features[2], st=int_features[3], insu=int_features[4], bmi=int_features[5], dpf=int_features[6], age=int_features[7])



@app.route('/heart')
def heart():
     return render_template('heart.html')
@app.route('/predict_heart', methods=['POST'])
def predict_heart():
    model=pickle.load(open('Model/model_heart.pkl','rb'))
    scaler = joblib.load('Model/model_scaler2.pkl')
    age=int(request.form['age'])
    sex=int(request.form['sex'])
    cp=int(request.form['cp'])
    trestbps=int(request.form['trestbps'])
    chol=int(request.form['chol'])
    fb=int(request.form['fb'])
    restecg=int(request.form['restecg'])
    exang=int( request.form['exang'])
    thalach=int(request.form['thalach'])
    oldpeak=int(request.form['oldpeak'])
    slope=float(request.form['slope'])
    ca=int( request.form['ca'])
    thal=float(request.form['thal'])
    int_features=[age, sex, cp, trestbps, chol, fb, restecg,
     exang, thalach, oldpeak, slope, ca, thal,]
    features = [np.array(int_features)]
    final_features = scaler.transform(features)
    prediction = model.predict(final_features)
    output = prediction
    print(prediction)
    #print("h")
    if(output==1):
       # prediction="you are suffering from heart disease"
        #print("you are not suffering from heart disease")
        return render_template('heart.html',prediction="you are suuffering from heart disease{}".format(output))
    else:
       # prediction="you are  not suffering from heart disease"
        return render_template('heart.html',prediction="you are not suffering from heart disease{}".format(output))
@app.route('/kidney')
def kidney():
    return render_template('kidney.html')
@app.route('/predict_kidney', methods=['POST'])
def predict_kidney():
    model=pickle.load(open('Model/model1_kidney.pkl','rb'))
    if request.method == 'POST':
        sg = float(request.form['sg'])
        htn = float(request.form['htn'])
        hemo = float(request.form['hemo'])
        dm = float(request.form['dm'])
        al = float(request.form['al'])
        appet = float(request.form['appet'])
        rc = float(request.form['rc'])
        pc = float(request.form['pc'])

        values = np.array([[sg, htn, hemo, dm, al, appet, rc, pc]])
        prediction = model.predict(values)
        if(prediction==1):
       
            return render_template('kidney.html',prediction="you are suffering from kidney disease{}".format(prediction))
        else:
       # prediction="you are  not suffering from heart disease"
            return render_template('kidney.html',prediction="you are not suffering from kidney disease{}".format(prediction))


@app.route('/parkinson')
def parkinson():
    return render_template('parkinson.html')
@app.route('/predict_pa', methods=['POST'])
def predict_pa():
    model=pickle.load(open('Model/model_pa.pkl','rb'))
    scaler = joblib.load('Model/model_scaler3.pkl')
    mdvp_fo=float(request.form['mdvp_fo'])
    mdvp_fhi=float(request.form['mdvp_fhi'])
    mdvp_flo=float(request.form['mdvp_flo'])
    mdvp_jitper=float(request.form['mdvp_jitper'])
    mdvp_jitabs=float(request.form['mdvp_jitabs'])
    mdvp_rap=float(request.form['mdvp_rap'])
    MDVP_PPQ=float(request.form['MDVP:PPQ'])
    Jitter_DDP=float(request.form['Jitter:DDP'])
    MDVP_Shimmer=float(request.form['MDVP:Shimmer'])
    MDVP_Shimmer=float(request.form['MDVP:Shimmer'])
    Shimmer_APQ3=float(request.form['Shimmer:APQ3'])
    Shimmer_APQ5=float(request.form['Shimmer:APQ5'])
    MDVP_APQ=float(request.form['MDVP:APQ'])
   # MDVP:APQ=int(request.form['MDVPAPQ'])
    Shimmer_DDA=float(request.form['Shimmer:DDA'])
    nhr=float(request.form['NHR'])
    hnr=float(request.form['HNR'])
    RPDE=float(request.form['RPDE'])
    DFA=float(request.form['DFA'])
    spread1=float(request.form['spread1'])
    spread2=float(request.form['spread2'])
    D2=float(request.form['spread3'])
    PPE=float(request.form['ppe'])
    int_features=[mdvp_fo,mdvp_fhi,mdvp_flo,mdvp_jitper,mdvp_jitabs,mdvp_rap,MDVP_PPQ,Jitter_DDP,MDVP_Shimmer,MDVP_Shimmer,
    Shimmer_APQ3,Shimmer_APQ5,MDVP_APQ,Shimmer_DDA,nhr,hnr,RPDE,DFA,spread1,spread2,D2,PPE]
    features = [np.array(int_features)]
    final_features = scaler.transform(features)
    prediction = model.predict(final_features)
    output = prediction
    print(prediction)
    #print("h")
    if(output==1):
       # prediction="you are suffering from heart disease"
        #print("you are not suffering from heart disease")
        return render_template('parkinson.html',prediction="you are suuffering from PARKINSON'S disease{}".format(output))
    else:
       # prediction="you are  not suffering from heart disease"
        return render_template('parkinson.html',prediction="you are not suffering from PARKINSON'S disease{}".format(output))


if __name__ == "__main__":
        app.run(debug=True)  