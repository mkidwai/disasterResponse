import sys
import re
import pandas as pd
import nltk
nltk.download('stopwords')
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import pickle
from sklearn.metrics import classification_report



def load_data(database_filepath):
    """
    Load data from database
    input:
        database file path
    outputs:
        category_names: categories names from the dataframe, X: text messages from the dataframe, Y: column names (categories)
    """
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)


    df = pd.read_sql_table('CategorizedMessages', engine)
    #X = df.message.values
    #y = df[df.columns[4:40]].values
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    """
    Split the text into list and remove unnessary elements 
    input:
        text 
    outputs:
        tokenized list of the given text
    """ 
   
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    

    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    """
    Build the model and fine tune the grid search 
    input:
        NONE 
    outputs:
        A model
    """ 
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__max_depth': [25, 100, 200],
#        'tfidf__use_idf': (True, False),
#        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__criterion': ['entropy', 'gini']
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the built in model
    input:
        mode: model to be tested. X_test, Y_test, category_names  
    outputs:
        NONE - print the classification report 
    """ 
    Y_pred = model.predict(X_test)

    for i in range(len(category_names)):
        print("Category: {:<25} \n {}".format(category_names[i], 
                                              classification_report(Y_test.iloc[:, i].values, Y_pred[:, i])))  



def save_model(model, model_filepath):
    """
    Save the model in the provided path 
    input:
        model, model_filepath along with the file name
    outputs:
        NONE - file get created
    """    
    pickle.dump(model, open(model_filepath, "wb"))

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