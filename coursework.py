"""Runnable versions of the CPSC 6300 exercises; see docs/methodology.md."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
from itertools import combinations
import argparse, hashlib, json, os, platform, time, urllib.request, zipfile

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVC
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, classification_report, confusion_matrix, silhouette_score
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

ROOT=Path(__file__).resolve().parent
SEED=42
_TF_CONFIGURED=False

def native(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    raise TypeError(type(value).__name__)

def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,default=native,allow_nan=False),encoding='utf-8')

def fetch(url,name):
    path=ROOT/'data/downloads'/name
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix(path.suffix+'.part')
        urllib.request.urlretrieve(url,temp)
        temp.replace(path)
    return path

def figure(out,name):
    plt.tight_layout();plt.savefig(out/name,dpi=160,bbox_inches='tight');plt.close()

def reg_metrics(y,pred):
    return {'mse':float(mean_squared_error(y,pred)),'rmse':float(np.sqrt(mean_squared_error(y,pred))),'mae':float(mean_absolute_error(y,pred)),'r2':float(r2_score(y,pred))}

def cls_metrics(y,pred):
    return {'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),'confusion_matrix':confusion_matrix(y,pred).tolist(),'report':classification_report(y,pred,output_dict=True,zero_division=0)}

def hw03(out,models,profile):
    names=['rating','review_count','isbn','booktype','author_url','year','genre_urls','dir','rating_count','name']
    df=pd.read_csv(ROOT/'data/hw03/goodreads.csv',header=None,names=names)
    before=len(df);missing=df.isna().sum().to_dict()
    for col in ['rating','review_count','year','rating_count']:df[col]=pd.to_numeric(df[col],errors='coerce')
    df=df.dropna(subset=['year','rating']).copy()
    df['author']=df.author_url.fillna('').str.rsplit('/',n=1).str[-1].str.replace(r'^\d+\.','',regex=True).str.replace('_',' ')
    df['genres']=df.genre_urls.fillna('').map(lambda x:'|'.join(part.rsplit('/',1)[-1] for part in str(x).split('|')))
    df.groupby('year').agg(books=('name','count'),mean_rating=('rating','mean')).to_csv(out/'year-summary.csv')
    df.to_csv(out/'cleaned-books.csv',index=False)
    df.rating.hist(bins=30,color='#cf9d3d');plt.xlabel('Book rating');plt.ylabel('Books');plt.title('Inspect the response distribution before modeling');figure(out,'ratings.png')
    return {'input_rows':before,'retained_rows':len(df),'missing_before':missing,'scope':'Data preparation and descriptive analysis; no model trained.'}

def hw04(out,models,profile):
    titanic=pd.read_csv(fetch('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv','titanic.csv'))
    summary={'titanic_rows':len(titanic),'missing_deck':int(titanic.deck.isna().sum()),'infants_under_3':int((titanic.age<3).sum()),'children_3_to_10':int(titanic.age.between(3,10).sum())}
    fig,axs=plt.subplots(1,2,figsize=(10,4));titanic.fare.plot.hist(bins=40,ax=axs[0],color='#d09e3d');axs[0].set(xlabel='Fare',ylabel='Passengers',title='Fare distribution')
    for label,group in titanic.groupby('survived'):axs[1].scatter(group.age,group.fare,s=10,alpha=.4,label='Survived' if label else 'Did not survive')
    axs[1].set(xlabel='Age',ylabel='Fare',title='Sample relationships');axs[1].legend();figure(out,'eda.png')
    # Restore the publicly available Cleveland exercise dataset, with source recorded.
    archive=fetch('https://archive.ics.uci.edu/static/public/45/heart+disease.zip','heart-disease.zip')
    with zipfile.ZipFile(archive) as z:
        name=next(n for n in z.namelist() if n.endswith('processed.cleveland.data'))
        import io
        heart=pd.read_csv(io.BytesIO(z.read(name)),header=None,na_values='?')
    heart.columns=['age','sex','cp','restbp','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']
    heart.describe().to_csv(out/'heart-description.csv')
    summary.update({'cleveland_rows':len(heart),'cleveland_source':'UCI Heart Disease public archive, processed.cleveland.data','interpretation':'Descriptions refer to the sampled records, not population care-seeking rates.','optional_missing_input':'The original mystery.csv visualization is archived only; its source data were not present in Drive.'})
    return summary

def hw05(out,models,profile):
    frame=pd.read_csv(ROOT/'data/hw05/multishapes.csv');X=StandardScaler().fit_transform(frame[['x','y']])
    results={};fig,axs=plt.subplots(1,3,figsize=(12,4))
    algorithms={'K-means':KMeans(n_clusters=3,random_state=109,n_init=10),'Ward':AgglomerativeClustering(n_clusters=3),'DBSCAN':DBSCAN(eps=.2)}
    for ax,(name,model) in zip(axs,algorithms.items()):
        labels=model.fit_predict(X);valid=labels>=0;clusters=len(set(labels[valid]))
        score=float(silhouette_score(X[valid],labels[valid],sample_size=min(2000,valid.sum()),random_state=SEED)) if clusters>1 else None
        results[name]={'clusters':clusters,'noise_rows':int((~valid).sum()),'silhouette_excluding_noise':score};ax.scatter(X[:,0],X[:,1],c=labels,s=5,cmap='tab10');ax.set_title(name);ax.set_xlabel('Standardized x');ax.set_ylabel('Standardized y')
        if hasattr(model,'predict'):joblib.dump(model,models/'shapes-kmeans.joblib')
    figure(out,'cluster-comparison.png')
    # Read the official MNIST NPZ directly: no cloned helper repository or shell commands.
    with np.load(fetch('https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz','mnist.npz'),allow_pickle=False) as d:digits=d['x_train'].reshape(-1,784).astype('float32')/255.
    n=1000 if profile=='preview' else 5000
    ids=np.random.default_rng(SEED).choice(len(digits),n,replace=False);digits=digits[ids]
    rows=[]
    for k in ([3,5,10] if profile=='preview' else range(2,20)):
        model=KMeans(n_clusters=k,n_init=5,max_iter=200,random_state=SEED).fit(digits)
        rows.append({'k':k,'inertia':float(model.inertia_),'silhouette':float(silhouette_score(digits,model.labels_,sample_size=min(n,1000),random_state=SEED))})
    pd.DataFrame(rows).to_csv(out/'digit-cluster-search.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(10,4));axs[0].plot([r['k'] for r in rows],[r['inertia'] for r in rows],'o-');axs[0].set(xlabel='k',ylabel='Within-cluster sum of squares');axs[1].plot([r['k'] for r in rows],[r['silhouette'] for r in rows],'o-');axs[1].set(xlabel='k',ylabel='Silhouette score');figure(out,'cluster-search.png')
    return {'shapes_rows':len(X),'shape_algorithms':results,'digit_rows':n,'digit_search':rows,'scope':'Unsupervised structure; silhouette is not classification accuracy. MNIST search uses a seeded subset and sampled silhouette for bounded compute.'}

def hw06(out,models,profile):
    columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
    train=pd.read_csv(ROOT/'data/hw06/boston_housing_train.csv',sep=r'\s+',header=None,names=columns)
    X=train.drop(columns='MEDV');y=train.MEDV
    xt,xv,yt,yv=train_test_split(X,y,test_size=.2,random_state=SEED)
    validation={}
    for name,features in [('subset',['RM','LSTAT','PTRATIO']),('all',list(X.columns))]:
        model=LinearRegression().fit(xt[features],yt);validation[name]=reg_metrics(yv,model.predict(xv[features]))
    candidates={'Mean baseline':DummyRegressor(),'Linear regression':LinearRegression(),'Ridge':make_pipeline(StandardScaler(),Ridge(alpha=1.)),'Lasso':make_pipeline(StandardScaler(),Lasso(alpha=.1,max_iter=10000)),'Random Forest':RandomForestRegressor(n_estimators=100,random_state=SEED,n_jobs=2),'Polynomial degree 2':make_pipeline(PolynomialFeatures(2,include_bias=False),LinearRegression())}
    kf=KFold(10,shuffle=True,random_state=SEED);cv={}
    for name,model in candidates.items():
        scores=cross_validate(model,X,y,cv=kf,scoring={'mse':'neg_mean_squared_error','r2':'r2'},n_jobs=1)
        cv[name]={'mean_mse':float(-scores['test_mse'].mean()),'std_mse':float(scores['test_mse'].std()),'mean_r2':float(scores['test_r2'].mean())}
    selected=min(cv,key=lambda n:cv[n]['mean_mse']);model=candidates[selected].fit(X,y)
    # Load the external test set only after selecting the model using development data.
    test=pd.read_csv(ROOT/'data/hw06/boston_housing_test.csv',sep=r'\s+',header=None,names=columns)
    pred=model.predict(test.drop(columns='MEDV'));metrics=reg_metrics(test.MEDV,pred)
    pd.DataFrame({'actual':test.MEDV,'predicted':pred}).to_csv(out/'test-predictions.csv',index=False);joblib.dump(model,models/'regression.joblib')
    fig,axs=plt.subplots(1,2,figsize=(11,4));names=list(cv);axs[0].barh(names,[cv[n]['mean_mse'] for n in names],color='#cc9e45');axs[0].set_xlabel('10-fold CV mean squared error');axs[1].scatter(test.MEDV,pred);lim=[min(test.MEDV.min(),pred.min()),max(test.MEDV.max(),pred.max())];axs[1].plot(lim,lim,'--',color='gray');axs[1].set(xlabel='Actual MEDV',ylabel='Predicted MEDV');figure(out,'regression.png')
    return {'development_rows':len(train),'test_rows':len(test),'validation':validation,'cross_validation':cv,'selection_rule':'Lowest mean MSE on the same shuffled 10-fold split for all candidates','selected_model':selected,'test':metrics,'objective':'Squared error for OLS/forest; L2 or L1 penalties for Ridge/Lasso.','scope':'Historical supplied classroom dataset, not an operational housing valuation model.'}

def hw07(out,models,profile):
    raw=pd.read_csv(ROOT/'data/hw07/Algerian_forest_fires_dataset_UPDATE-1.csv');raw.columns=raw.columns.str.strip()
    ytext=raw.Classes.astype(str).str.strip();valid=ytext.isin(['fire','not fire']);df=raw.loc[valid].copy()
    X=df.drop(columns=['Classes','day','month','year'],errors='ignore').apply(pd.to_numeric,errors='coerce');good=X.notna().all(axis=1);X=X.loc[good];y=(ytext.loc[X.index]=='fire').astype(int)
    xt,xe,yt,ye=train_test_split(X,y,test_size=.3,stratify=y,random_state=SEED)
    candidates={'Majority baseline':DummyClassifier(strategy='most_frequent'),'Logistic regression':make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)),'Linear SVC':make_pipeline(StandardScaler(),LinearSVC(max_iter=10000,random_state=SEED)),'Ridge classifier':make_pipeline(StandardScaler(),RidgeClassifier())}
    cv={};folds=StratifiedKFold(5,shuffle=True,random_state=SEED)
    for name,model in candidates.items():
        score=cross_validate(model,xt,yt,cv=folds,scoring={'accuracy':'accuracy','macro_f1':'f1_macro'},n_jobs=1)
        cv[name]={'accuracy':float(score['test_accuracy'].mean()),'macro_f1':float(score['test_macro_f1'].mean())}
    selected=max(cv,key=lambda n:cv[n]['macro_f1']);model=candidates[selected].fit(xt,yt);pred=model.predict(xe);metrics=cls_metrics(ye,pred);joblib.dump(model,models/'fire-classifier.joblib')
    # shares is a count: use a regression-appropriate selector rather than treating counts as classes.
    from sklearn.feature_selection import SelectKBest,f_regression
    news=pd.read_csv(ROOT/'data/hw07/OnlineNewsPopularity.csv');news.columns=news.columns.str.strip();news=news.drop(columns=['url'],errors='ignore')
    nx=news.drop(columns='shares');ny=news.shares
    nx,_,ny,_=train_test_split(nx,ny,test_size=.3,random_state=50)
    selection=SelectKBest(f_regression,k=10).fit(nx,ny)
    top=pd.DataFrame({'feature':nx.columns[selection.get_support()],'f_score':selection.scores_[selection.get_support()]}).sort_values('f_score',ascending=False);top.to_csv(out/'news-feature-ranking.csv',index=False)
    plt.imshow(metrics['confusion_matrix'],cmap='Blues');plt.xticks([0,1],['Not fire','Fire']);plt.yticks([0,1],['Not fire','Fire']);plt.xlabel('Predicted');plt.ylabel('Actual');plt.title(selected)
    for (i,j),v in np.ndenumerate(np.array(metrics['confusion_matrix'])):plt.text(j,i,str(v),ha='center',va='center')
    figure(out,'classification.png')
    return {'retained_rows':len(X),'train_rows':len(xt),'test_rows':len(xe),'cross_validation':cv,'selected_model':selected,'selection_rule':'Training-only 5-fold macro F1','test':metrics,'news_selected_features':top.feature.tolist(),'news_selector':'F regression on training rows only; replaces chi-square on raw share counts.','objectives':{'logistic':'Binary log loss + L2 penalty','svc':'Squared hinge loss + L2 penalty','ridge':'Squared loss + L2 penalty'},'scope':'Classification of recorded daily conditions, not a validated future-fire forecasting system.'}

def tensorflow():
    global _TF_CONFIGURED
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)
    if not _TF_CONFIGURED:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        _TF_CONFIGURED=True
    return tf

def tf_dataset(tf,X,y,batch=32,shuffle=False):
    ds=tf.data.Dataset.from_tensor_slices((X,y))
    if shuffle:ds=ds.shuffle(min(len(X),10000),seed=SEED,reshuffle_each_iteration=True)
    opts=tf.data.Options();opts.threading.private_threadpool_size=2;opts.threading.max_intra_op_parallelism=2
    return ds.batch(batch).with_options(opts)

def hw08(out,models,profile):
    tf=tensorflow();data=ROOT/'data/hw08'
    X=np.load(data/'sinewaves_X_train.npy',allow_pickle=False).astype('float32');y=np.load(data/'sinewaves_y_train.npy',allow_pickle=False).astype('float32')
    xt,xv,yt,yv=X[:8000],X[8000:],y[:8000],y[8000:]
    if profile=='preview':xt,yt=xt[:1000],yt[:1000]
    model=tf.keras.Sequential([tf.keras.Input((1000,)),tf.keras.layers.Dense(128,activation='relu'),tf.keras.layers.Dense(64,activation='relu'),tf.keras.layers.Dense(32,activation='relu'),tf.keras.layers.Dense(4)])
    model.compile(loss='mean_absolute_error',optimizer='adam');epochs=2 if profile=='preview' else 50
    history=model.fit(tf_dataset(tf,xt,yt,shuffle=True),validation_data=tf_dataset(tf,xv,yv),epochs=epochs,verbose=2)
    results={}
    for condition,suffix in [('in_distribution','test'),('extended','extended_test')]:
        xe=np.load(data/f'sinewaves_X_{suffix}.npy',allow_pickle=False).astype('float32');ye=np.load(data/f'sinewaves_y_{suffix}.npy',allow_pickle=False).astype('float32')
        pred=model.predict(tf_dataset(tf,xe,ye),verbose=0);results[condition]={'rows':len(xe),'parameter_mae':float(np.abs(ye-pred).mean()),'per_parameter_mae':np.abs(ye-pred).mean(axis=0).tolist()}
        pd.DataFrame(np.column_stack([ye,pred]),columns=[f'true_{n}' for n in 'abcd']+[f'pred_{n}' for n in 'abcd']).to_csv(out/f'{condition}-predictions.csv',index=False)
        points=np.linspace(0,10*np.pi,1000);fig,axs=plt.subplots(1,2,figsize=(11,3.7))
        for ax,i in zip(axs,[0,1]):
            a,b,c,d=pred[i];ax.plot(points,xe[i],label='Observed waveform');ax.plot(points,a*np.sin(b*points)+c*np.cos(d*points),'--',label='Implied by prediction');ax.set(xlabel='x',ylabel='Amplitude');ax.legend(fontsize=8)
        figure(out,f'{condition}-curves.png')
    plt.plot(history.history['loss'],label='Training MAE');plt.plot(history.history['val_loss'],label='Validation MAE');plt.xlabel('Epoch');plt.ylabel('Parameter MAE');plt.legend();figure(out,'learning-curve.png')
    model.save(models/'sinewave.keras');write_json(out/'history.json',history.history)
    return {'train_rows':len(xt),'validation_rows':len(xv),'epochs':epochs,'batch_size':32,'inputs':1000,'outputs':['a','b','c','d'],'first_layer_parameters':model.layers[0].count_params(),'total_parameters':model.count_params(),'architecture':[1000,128,64,32,4],'objective':'Mean absolute error on four generating parameters','optimizer':'Adam','evaluation':results,'direction':'1000 waveform samples -> four parameters; this is not a general Fourier transform.'}

def hw09(out,models,profile):
    tf=tensorflow();keras=tf.keras
    (X,y),(xe,ye)=keras.datasets.cifar100.load_data(label_mode='fine');y=y.ravel();ye=ye.ravel()
    X=X.astype('float32')/255;xe=xe.astype('float32')/255
    xt,xv,yt,yv=train_test_split(X,y,test_size=.1,stratify=y,random_state=SEED)
    if profile=='preview':xt,yt=xt[:2000],yt[:2000];xv,yv=xv[:1000],yv[:1000];xe,ye=xe[:1000],ye[:1000]
    model=keras.Sequential([keras.Input((32,32,3)),keras.layers.Conv2D(32,3,activation='relu',padding='same'),keras.layers.Conv2D(32,3,activation='relu'),keras.layers.MaxPooling2D(),keras.layers.Dropout(.25),keras.layers.Conv2D(64,3,activation='relu',padding='same'),keras.layers.Conv2D(64,3,activation='relu'),keras.layers.MaxPooling2D(),keras.layers.Dropout(.25),keras.layers.Flatten(),keras.layers.Dense(512,activation='relu'),keras.layers.Dropout(.5),keras.layers.Dense(100,activation='softmax')])
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy']);epochs=1 if profile=='preview' else 5
    hist=model.fit(tf_dataset(tf,xt,yt,64,True),validation_data=tf_dataset(tf,xv,yv,64),epochs=epochs,verbose=2)
    pred=model.predict(tf_dataset(tf,xe,ye,64),verbose=0).argmax(axis=1);cnn=cls_metrics(ye,pred);model.save(models/'cifar100-cnn.keras');write_json(out/'cnn-history.json',hist.history)
    del X,xt,xv,model;keras.backend.clear_session();keras.utils.set_random_seed(SEED)
    (X,y),(xe,ye)=keras.datasets.reuters.load_data(num_words=10000,test_split=.2,seed=SEED)
    X=keras.utils.pad_sequences(X,maxlen=200);xe=keras.utils.pad_sequences(xe,maxlen=200)
    # Small classes preclude a stratified validation split; record a seeded random split.
    xt,xv,yt,yv=train_test_split(X,y,test_size=.1,random_state=SEED)
    if profile=='preview':xt,yt=xt[:1000],yt[:1000];xv,yv=xv[:500],yv[:500];xe,ye=xe[:500],ye[:500]
    model=keras.Sequential([keras.Input((200,),dtype='int32'),keras.layers.Embedding(10000,128),keras.layers.LSTM(128,dropout=.2,recurrent_dropout=.2),keras.layers.Dense(46,activation='softmax')])
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy']);hist=model.fit(tf_dataset(tf,xt,yt,64,True),validation_data=tf_dataset(tf,xv,yv,64),epochs=epochs,verbose=2)
    pred=model.predict(tf_dataset(tf,xe,ye,64),verbose=0).argmax(axis=1);lstm=cls_metrics(ye,pred);model.save(models/'reuters-lstm.keras');write_json(out/'lstm-history.json',hist.history)
    plt.bar(['CIFAR-100 CNN','Reuters LSTM'],[cnn['accuracy'],lstm['accuracy']],color=['#d1a343','#779cbb']);plt.ylim(0,1);plt.ylabel('Accuracy');plt.title(f'{profile.title()} run: separate datasets, not a model ranking');figure(out,'neural-tasks.png')
    return {'epochs':epochs,'cnn':{'test_rows':1000 if profile=='preview' else 10000,'metrics':cnn,'validation':'Held out from CIFAR training data, not the test data.'},'lstm':{'test_rows':len(ye),'metrics':lstm,'vocabulary':10000,'max_sequence_length':200},'objective':'Categorical cross-entropy; integer labels use the equivalent sparse formulation.','scope':'Different tasks and label spaces; accuracies are not a head-to-head comparison.'}

def hw10(out,models,profile):
    os.environ['USE_TF']='0'
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer,AutoModelForSequenceClassification,DataCollatorWithPadding,Trainer,TrainingArguments,set_seed
    torch.set_num_threads(2);set_seed(SEED)
    data=ROOT/'data/hw10/alt_syllogisms.csv'
    if not data.exists():
        import gdown
        data.parent.mkdir(exist_ok=True);gdown.download(id='19FYxTXMop51l6aBpdAOoq-LZiSOuK6Qn',output=str(data),quiet=False)
    df=pd.read_csv(data);df=df[['Text','Valid']].dropna().copy();df.Valid=df.Valid.astype(str).str.strip().str.lower().map({'yes':1,'no':0,'true':1,'false':0,'1':1,'0':0})
    if df.Valid.isna().any():raise ValueError('Unrecognized validity label; inspect the input.')
    df.Valid=df.Valid.astype(int)
    # Keep identical wording out of different partitions. This does not establish generalization to unseen logical forms.
    conflicts=df.groupby('Text').Valid.nunique()
    if (conflicts>1).any():raise ValueError('Identical text has conflicting labels.')
    before=len(df);df=df.drop_duplicates('Text')
    development,test=train_test_split(df,test_size=.2,random_state=SEED,stratify=df.Valid)
    train,val=train_test_split(development,test_size=.2,random_state=SEED,stratify=development.Valid)
    full_train_rows=len(train)
    if profile=='preview':train=train.groupby('Valid',group_keys=False).head(16);val=val.groupby('Valid',group_keys=False).head(8)
    metrics={};baseline=DummyClassifier(strategy='most_frequent').fit(np.zeros((len(train),1)),train.Valid)
    metrics['majority_baseline']=cls_metrics(test.Valid,baseline.predict(np.zeros((len(test),1))))
    for name in ['facebook/bart-base','FacebookAI/roberta-base']:
        set_seed(SEED);tokenizer=AutoTokenizer.from_pretrained(name,revision='main')
        model=AutoModelForSequenceClassification.from_pretrained(name,num_labels=2)
        def encode(batch):
            enc=tokenizer(batch['Text'],truncation=True,max_length=256);enc['labels']=batch['Valid'];return enc
        def dataset(frame):return Dataset.from_pandas(frame.reset_index(drop=True),preserve_index=False).map(encode,batched=True,remove_columns=['Text','Valid'])
        ds_train,ds_val,ds_test=dataset(train),dataset(val),dataset(test)
        dest=models/('transformer-'+name.split('/')[-1])
        args=TrainingArguments(output_dir=str(out/('training-'+name.split('/')[-1])),learning_rate=2e-5,per_device_train_batch_size=2 if profile=='preview' else 16,per_device_eval_batch_size=8,num_train_epochs=1 if profile=='preview' else 3,max_steps=2 if profile=='preview' else -1,weight_decay=.01,eval_strategy='epoch',save_strategy='no',report_to=[],seed=SEED,dataloader_num_workers=0,use_cpu=not torch.cuda.is_available())
        trainer=Trainer(model=model,args=args,train_dataset=ds_train,eval_dataset=ds_val,data_collator=DataCollatorWithPadding(tokenizer),processing_class=tokenizer)
        trainer.train();prediction=trainer.predict(ds_test);logits=prediction.predictions
        if isinstance(logits,tuple):logits=logits[0]
        predicted=logits.argmax(axis=1);metrics[name]=cls_metrics(test.Valid,predicted);metrics[name]['pretrained_commit']=model.config._commit_hash
        trainer.save_model(dest);tokenizer.save_pretrained(dest)
        pd.DataFrame({'actual':test.Valid.to_numpy(),'prediction':predicted}).to_csv(out/(name.split('/')[-1]+'-predictions.csv'),index=False)
        del trainer,model;import gc;gc.collect()
    return {'input_rows':before,'deduplicated_rows':len(df),'train_rows':len(train),'full_train_rows':full_train_rows,'validation_rows':len(val),'test_rows':len(test),'test_prevalence':float(test.Valid.mean()),'steps_per_model':2 if profile=='preview' else None,'metrics':metrics,'objective':'Two-class cross-entropy; pretrained weights fine-tuned with AdamW.','scope':'Preview is a two-step execution demonstration, not evidence of trained model quality. Text deduplication does not remove logical-template overlap.'}

def hw11(out,models,profile):
    data=pd.read_csv(ROOT/'data/hw11/groceries-groceries.csv');columns=[c for c in data if 'Item' in c]
    chosen={'bottled beer','bottled water','brandy','brown bread','butter','syrup','sweet spreads','beverages','berries','beef','bathroom cleaner','baking powder','bags','baby food','baby cosmetics','Instant food products','tea','toilet cleaner','vinegar','waffles','whisky','white bread','white wine','yogurt','zwieback','whole milk','whipped/sour cream','abrasive cleaner'}
    baskets=[set(row.dropna().astype(str))&chosen for _,row in data[columns].iterrows()];n=len(baskets);min_count=max(1,int(np.ceil(.0001*n)))
    support={};previous=set()
    # Apriori pruning: a candidate survives only if every (k-1)-subset was frequent.
    for k in range(1,4):
        counts=Counter()
        for basket in baskets:
            for combination in combinations(sorted(basket),k):
                key=frozenset(combination)
                if k==1 or all(frozenset(sub) in previous for sub in combinations(combination,k-1)):counts[key]+=1
        previous={key for key,count in counts.items() if count>=min_count}
        support.update({key:counts[key]/n for key in previous})
    rows=[]
    for items,value in support.items():
        if len(items)<2:continue
        for k in range(1,len(items)):
            for a in combinations(sorted(items),k):
                a=frozenset(a);b=items-a;confidence=value/support[a];lift=confidence/support[b]
                if confidence>=.8:rows.append({'antecedent':' | '.join(sorted(a)),'consequent':' | '.join(sorted(b)),'support':value,'confidence':confidence,'lift':lift,'leverage':value-support[a]*support[b]})
    rules=pd.DataFrame(rows,columns=['antecedent','consequent','support','confidence','lift','leverage']).sort_values(['lift','support'],ascending=False)
    rules.to_csv(out/'association-rules.csv',index=False);filtered=rules[(rules.antecedent=='baby food')&(rules.confidence>.85)&(rules.lift>1)];filtered.to_csv(out/'baby-food-rules.csv',index=False)
    if len(rules):plt.scatter(rules.support,rules.confidence,c=rules.lift,cmap='viridis');plt.colorbar(label='Lift')
    plt.xlabel('Support');plt.ylabel('Confidence');plt.title('Association rules: co-occurrence, not causation');figure(out,'association-rules.png')
    return {'transactions':n,'candidate_items':len(chosen),'minimum_support':.0001,'max_itemset_size':3,'frequent_itemsets':len(support),'rules_confidence_at_least_08':len(rules),'baby_food_rules_after_filter':len(filtered),'objective':'Frequent-itemset counting with Apriori pruning, not supervised loss minimization.','lift_definition':'P(B|A) / P(B)','scope':'Low-support rules can be unstable; no causal or commercial impact is established.'}

def hw12(out,models,profile):
    data=pd.read_csv(ROOT/'data/hw12/wines.csv');X=data.drop(columns='quality');y=pd.cut(data.quality,[0,5,7,10],labels=[0,1,2]).astype(int)
    xt,xe,yt,ye=train_test_split(X,y,test_size=.2,stratify=y,random_state=8)
    candidates={'Majority baseline':DummyClassifier(strategy='most_frequent'),'All features':make_pipeline(StandardScaler(),LogisticRegression(max_iter=10000)),'PCA 2':make_pipeline(StandardScaler(),PCA(2),LogisticRegression(max_iter=10000)),'PCA 10':make_pipeline(StandardScaler(),PCA(10),LogisticRegression(max_iter=10000)),'SVD 10':make_pipeline(StandardScaler(),TruncatedSVD(10,random_state=SEED),LogisticRegression(max_iter=10000))}
    folds=StratifiedKFold(5,shuffle=True,random_state=SEED);cv={}
    for name,model in candidates.items():
        score=cross_validate(model,xt,yt,cv=folds,scoring={'accuracy':'accuracy','macro_f1':'f1_macro'},n_jobs=1);cv[name]={'accuracy':float(score['test_accuracy'].mean()),'macro_f1':float(score['test_macro_f1'].mean())}
    selected=max(cv,key=lambda n:cv[n]['macro_f1']);model=candidates[selected].fit(xt,yt);metrics=cls_metrics(ye,model.predict(xe));joblib.dump(model,models/'wine-pipeline.joblib')
    scaler=StandardScaler().fit(xt);pca=PCA().fit(scaler.transform(xt));variance=pca.explained_variance_ratio_
    plt.plot(range(1,len(variance)+1),variance.cumsum(),'o-');plt.xlabel('Principal components');plt.ylabel('Cumulative explained variance');plt.ylim(0,1.02);figure(out,'pca-variance.png')
    return {'rows':len(data),'train_rows':len(xt),'test_rows':len(xe),'cross_validation':cv,'selected_model':selected,'selection_rule':'Training-only 5-fold macro F1','test':metrics,'pca_explained_variance_ratio':variance.tolist(),'objectives':'PCA maximizes retained variance; classification uses multinomial log loss.','scope':'Explained variance is not predictive accuracy; scaling and reduction are fitted inside each training fold.'}

TASKS={f'hw{i:02}':globals()[f'hw{i:02}'] for i in range(3,13)}

def run_task(task,profile='full',output_root=None):
    if task not in TASKS:raise ValueError(f'Unknown task {task}')
    if profile not in ['full','preview']:raise ValueError(profile)
    out=(Path(output_root) if output_root else ROOT/'reports/runs')/profile/task;out.mkdir(parents=True,exist_ok=True)
    models=ROOT/'models'/profile/task;models.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();print(f'Running {task} ({profile})',flush=True)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):result=TASKS[task](out,models,profile)
    import importlib.metadata
    versions={}
    for package in ['numpy','pandas','scipy','scikit-learn','matplotlib','tensorflow','keras','torch','transformers','datasets']:
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:pass
    result.update({'task':task,'profile':profile,'run_at_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.perf_counter()-started,'seed':SEED,'python':platform.python_version(),'dependencies':versions,'platform':platform.platform(),'cpu_threads':2,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    write_json(out/'metrics.json',result);print(json.dumps(result,default=native)[:1800],flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('task',choices=[*TASKS,'classical','all']);parser.add_argument('--profile',choices=['full','preview'],default='full');parser.add_argument('--output-root',type=Path)
    args=parser.parse_args();tasks=list(TASKS) if args.task=='all' else ['hw03','hw04','hw05','hw06','hw07','hw11','hw12'] if args.task=='classical' else [args.task]
    for task in tasks:run_task(task,args.profile,args.output_root)
