import sys
import pandas as pd
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    df_merge = messages.merge(categories, how='outer',\
                               on=['id'])

    df = df_merge.sort_values(['id'])
    return df
    


def clean_data(df):
    categories = df['categories'].str.split(";",36,expand = True)
    row = categories.head(1)

    category_colnames = row.apply(lambda x: x.str.slice(0,(len(x)-3)))
    categories.columns = category_colnames.transpose()[0]

    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
        categories[column] = categories[column].astype(int)


    df.drop(['categories'], axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], sort=False, axis=1)

    #duplicateList = df.duplicated()
    #print(sum(duplicateList))

    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filename):
    #engine = create_engine('sqlite:///data/YourDatabaseName.db')
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('CategorizedMessages', engine, index=False)
      


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        
        print(df.head())

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()