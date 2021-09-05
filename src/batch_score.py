import pandas as pd
import numpy as np
import sys
sys.path.insert(1, '/src')
import train
import evaluate
import batch_prepare
from subprocess import call
import json
import ast
import os

def main():
    ### check the file contents of the batch_values.txt, if its length is empty, it means no user input has been given.
    ### in that case we use test dataset to score our model
    with open('data/batch_values.txt', 'r') as f:
        if len(f.readlines())==0:
            ### if batch_values.txt is empty, batch is created using test dataset
            _,test_X,test_y = train.main()
            BATCH_SIZE = 10
            batch_test_y = test_y.iloc[-BATCH_SIZE:]
        else:
            ### if new batch data is provided its corresponding true labels should be saved in batch_true_labels.txt file  
            with open('data/batch_true_labels.txt', 'r') as file:
                try:
                    batch_test_y = np.array(ast.literal_eval(file.readlines()[0]))
                except IndexError:
                    print("Error: Add actual target labels for the batch in file data/batch_true_labels.txt")
                    return
    
    ### import prediction values from the file batch_score_predictions.txt 
    with open('data/batch_score_predictions.txt') as f:
        try:
            pred_y = np.array(f.readlines())[0]
            pred_y = json.loads(pred_y)
        except IndexError:
            print("run make_predictions.sh script to get prediction values for the batch")
            return    
    
    ### evaluate model performance
    acc_score, f1Score, tn, fp, fn, tp = evaluate.eval_metrics(batch_test_y, pred_y)
    print("Accuracy Score: ", acc_score)
    print("F1 Score: ", f1Score)
    print("TN FP FN TP: ", tn, fp, fn, tp)
    

if __name__ == "__main__":
    main()

