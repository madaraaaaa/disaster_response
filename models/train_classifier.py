import sys
import nltk
from nltk.tokenize import word_tokenize
nltk.download('words')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
# import libraries
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import re
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import multioutput
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
nltk.download('stopwords')
from sklearn.metrics import fbeta_score, make_scorer
import pickle

import matplotlib.pyplot as p
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

def load_data(database_filepath, table_name = 'Table'):
    pass
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table(table_name,engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names=list(Y.columns)
    return X,Y,category_names


def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    stemmed = [PorterStemmer().stem(w) for w in words]
   
    
    
    
    return stemmed


def build_model():
    new_pipeline = pipeline =  Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier (RandomForestClassifier()))
        ])
    parameters = {
         'vect__max_df': (0.75, 1.0),
           
                #'clf__estimator__n_estimators': [10, 20],
                #'clf__estimator__min_samples_split': [2, 5],
             
    }
    
    cv =GridSearchCV(new_pipeline,  param_grid=parameters)             
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    predict=model.predict(X_test)
    df_p=pd.DataFrame(predict, columns=category_names)
    for i in df_p.columns:
          print(classification_report(Y_test[i], df_p[i]))
          print("            ")
    
        
    
  


def save_model(model, model_filepath):
    model_pkl = open(model_filepath, 'wb')
    pickle.dump(model, model_pkl)
    model_pkl.close()



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()