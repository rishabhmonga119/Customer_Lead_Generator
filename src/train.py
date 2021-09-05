"""
**The Machine Learning Development**
"""
import os
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(1, '/src')
import data_processing
from sklearn.ensemble import RandomForestClassifier

def main():
    """
    The ML training steps, including:
    1. Import preprocessed data
    2. Split train and test sets
    3. Train a RandomForest classifier
    4. fit the model
    5. Return model and test dataset
    """
    X,y = data_processing.main()
    train_X,test_X, train_y, test_y = train_test_split(X, y, test_size=0.15, random_state=42)
    model = RandomForestClassifier()
    model.fit(train_X, train_y)
    return model, test_X, test_y

if __name__ == "__main__":
    main()
