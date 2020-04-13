# Project 2 - Disaster Response Pipelines
Rubric: https://review.udacity.com/#!/rubrics/1565/view


## Project Summary

Following a disaster there will be millions and millions of communications, and the goal is to identify which are relevant to the disaster response professionals (Ex. Ambulances, Police, ...).

Data source is 'Figure Eight' and includes theirs pre-labeled tweets and text messages from real life disasters.


## Project Components

There are three components on this project.

1. ETL Pipeline (process_data.py)
A data cleaning pipeline that:

Loads the messages and categories datasets
Merges the two datasets
Cleans the data
Stores it in a SQLite database


2. ML Pipeline (train_classifier.py)
A machine learning pipeline that:

Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file

3. Flask Web App
The app that will generate a webpage to acess the application.


## Instructions to Run:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
