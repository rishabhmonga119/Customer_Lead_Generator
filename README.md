The given case study was to analyse the file CustomerData_LeadGenerator.csv, identify training data and labels within the dataset, and train and deploy a machine learning model. On closely examining the data, I chose to use b_gekauft_gesamt as my target label and the other features (except the feature fakeID) as the training data. 

Prerequistes to start the MLFlow server:
1. Download and install conda (https://docs.conda.io/en/latest/miniconda.html)

Steps to start the prediction server and make predictions:
1. Run command:
```
conda env create --file conda.yaml
```
2. Activate conda environment:
```
conda activate RandomForest
```
3. Run command:
```
mlflow models serve -m mlruns/0/8ad24d33836340df9a724d04ef972154/artifacts/model -p 1234
```
4. Add the batch data in the batch_data.csv file within the data folder. It should contain all the columns as in CustomerData_LeadGenerator.csv, except the target label b_gekauft_gesamt and the columns fakeID. If no data is given in batch_data.csv file, the last 10 rows of the test dataset is used as a batch to make predictions. 

5. For the batch scoring, add the true labels for the batch in batch_true_labels.txt as a list. 

6. Run the shell script make_predictions.sh 
```
sh make_predictions.sh
```