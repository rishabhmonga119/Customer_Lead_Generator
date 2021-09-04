import pandas as pd
import numpy as np
import sys
sys.path.insert(1, '/src')
import train
import evaluate
import subprocess
import json

def create_batch(test_X,test_y,BATCH_SIZE):
    batch = test_X.iloc[-BATCH_SIZE:].values.tolist()
    batch_test = test_y.iloc[-BATCH_SIZE:]
    batch_data = []
    ## remove white spaces in the list because predtion server expects no spaces between elements
    for i in batch:
        i = str(i)
        i = i.replace(' ','')
        batch_data.append(i)
    with open('data/batch_values.txt', 'w') as f:
        f.write(','.join(str(i) for i in batch_data))
    return batch_test

def main():
    with open('data/batch_values.txt', 'w') as f:
        print(f)
    _,test_X,test_y = train.main()
    BATCH_SIZE = 10
    batch_test = create_batch(test_X, test_y, BATCH_SIZE)
    #subprocess.call(['sh', 'make_predictions.sh'])
    with open('data/batch_score_predictions.txt') as f:
        pred_y = np.array(f.readlines())[0]
        pred_y = json.loads(pred_y)
    acc_score, f1Score, tn, fp, fn, tp = evaluate.eval_metrics(batch_test, pred_y)
    print(acc_score)
    print(f1Score)
    print(tn, fp, fn, tp)
    

if __name__ == "__main__":
    main()

