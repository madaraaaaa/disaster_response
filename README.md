# disaster-response

build a model for an API that classifies disaster messages. Using the web app an emergency worker can input a new message and get classification results in several categories so to have an idea what kind of help is needed Install
## libraries:

NumPy
Pandas
Matplotlib
Json
Plotly
Nltk
Flask
Sklearn
Sqlalchemy
Sys
Re
Pickle

### process_data.py: this file contain all the function need to extract the data and clean the data
### train_classifier.py: contain all the function to build and pipline the model and also save the model
### ETL Pipeline Preparation.ipynb: conatin the code for anylsis the data
### ML Pipeline Preparation.ipynb: bulid the machine learning model
### disaster_messages.csv, disaster_categories.csv contain sample messages (real messages that were sent during disaster events) and categories datasets in csv format.

## templates folder: This folder contains all of the files necessary to run and render the web app.
## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`


2. Run the following command in the app's directory to run your web app.
    `python run.py`
