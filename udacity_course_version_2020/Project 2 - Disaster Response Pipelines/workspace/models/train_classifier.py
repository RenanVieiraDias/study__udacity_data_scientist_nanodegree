import pandas as pd
import re
import sqlalchemy as sql
import sys
import pickle
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl


def load_data(database_filepath):
    # load data from database
    engine = sql.create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("messages",con=engine)

    # Defining the columns to get Y
    y_columns = list(df.columns)
    y_columns.remove('id')
    y_columns.remove('message')
    y_columns.remove('original')
    y_columns.remove('genre')

    Y = df[y_columns].values
    X = df['message'].values
    return X, Y, y_columns


def tokenize(text):
    tokens = nltk.tokenize.word_tokenize(text)
    lemmatizer = nltk.stem.WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return(clean_tokens)


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier(max_depth=7)))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    for i in range(0,len(category_names)):
        print( '------------------------------------------' )
        print( category_names[i] )
        print( classification_report(Y_test[:,i], Y_pred[:,i], target_names=['class 0 - false','class 1 - true ']) )


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


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