import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import sys
sys.path.insert(1, '/src')
import utils

def data_cleaning(input_data):
    """ 
    The data cleaning converts all the columns as type float. It drops rows with missing values or values not type convertable to 
    float. It renames columns to easy to use column names.

    :param input_data: Input data
    :type:df

    :return input_data: Cleaned data
    :rtype:df """

    ### ensuring all the training data is type float. Missing values/strings will be replaced with None and dropped from the dataframe 
    for i in range(len(input_data.columns)):
        try:
            input_data.iloc[:,i] = input_data.iloc[:,i].apply(utils.type_conversion)
        except ValueError:
            return None
    input_data.dropna(inplace=True)
    
    ### renaming columns for easy handling
    for i in range(1,len(input_data.columns)-1):  
        input_data_cleaned = input_data.rename(columns={input_data.columns[i]: 'param'+str(i)})

    return input_data_cleaned

def data_balancing(input_data_cleaned):
    """ The data is divided into training data (X) and labels (y). 
    During the EDA stage there was class imbalance seen between the label 1 and 0. 
    In order to tackle class imbalance, SMOTE method is used to oversample the minority class 
    and RandomUndersampler is used to undersample the majority class 

    :param input_data_cleaned: Cleaned data 
    :type:df

    :return X,y: Transformed data containing the training data.
    :rtype:df """
    ### ID column is dropped for learning a classifier
    X = input_data_cleaned.iloc[:,1:-1]
    y = input_data_cleaned.iloc[:,-1]

    ### Oversampling of minority class and undersampling of majority class 
    oversample = SMOTE()
    undersample = RandomUnderSampler()
    steps = [("o", oversample), ("u", undersample)]
    pipeline = Pipeline(steps=steps)
    X,y = pipeline.fit_resample(X,y)
    return X,y

def main():
    input_data = pd.read_csv('data/CustomerData_LeadGenerator.csv')
    input_data_cleaned = data_cleaning(input_data)
    X,y = data_balancing(input_data_cleaned)
    return X,y


if __name__ == "__main__":
    main()
