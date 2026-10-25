"""Load a saved model and inspect one documented example without retraining."""
from pathlib import Path
import argparse,json,os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import numpy as np
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parent

def predict(task,row=0,profile='full'):
    if task=='hw08':
        import tensorflow as tf
        model=tf.keras.models.load_model(ROOT/f'models/{profile}/hw08/sinewave.keras')
        X=np.load(ROOT/'data/hw08/sinewaves_X_test.npy',allow_pickle=False)
        y=np.load(ROOT/'data/hw08/sinewaves_y_test.npy',allow_pickle=False)
        pred=model(X[row:row+1].astype('float32'),training=False).numpy()[0]
        return {'task':task,'row':row,'parameters':['a','b','c','d'],'actual':y[row].tolist(),'prediction':pred.tolist(),'parameter_mae':float(np.abs(y[row]-pred).mean())}
    if task=='hw06':
        columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
        df=pd.read_csv(ROOT/'data/hw06/boston_housing_test.csv',sep=r'\s+',header=None,names=columns)
        model=joblib.load(ROOT/f'models/{profile}/hw06/regression.joblib')
        return {'task':task,'row':row,'actual':float(df.iloc[row].MEDV),'prediction':float(model.predict(df.drop(columns='MEDV').iloc[row:row+1])[0]),'target_units':'MEDV in $1000s, historical course data'}
    if task=='hw12':
        df=pd.read_csv(ROOT/'data/hw12/wines.csv');model=joblib.load(ROOT/f'models/{profile}/hw12/wine-pipeline.joblib')
        actual=0 if df.iloc[row].quality<=5 else 1 if df.iloc[row].quality<=7 else 2
        return {'task':task,'row':row,'actual_category':actual,'predicted_category':int(model.predict(df.drop(columns='quality').iloc[row:row+1])[0]),'categories':{'0':'quality <= 5','1':'quality 6 or 7','2':'quality >= 8'},'scope':'Illustrative source row; use saved test predictions for held-out evaluation.'}
    raise ValueError('Supported saved-model examples: hw06, hw08, hw12')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('task',choices=['hw06','hw08','hw12']);parser.add_argument('--row',type=int,default=0);parser.add_argument('--profile',choices=['full','preview'],default='full')
    args=parser.parse_args();print(json.dumps(predict(args.task,args.row,args.profile),indent=2))
