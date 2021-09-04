The given case study was to analyse the file CustomerData_LeadGenerator.csv and identify training data and labels within the dataset. On examining the data, I chose to use b_gekauft_gesamt as the target label and the other features as the training data. 

Prerequistes to start the MLFlow server:
1. Conda env. (https://docs.conda.io/en/latest/miniconda.html)

Steps to start the prediction server:
1. Run command:
```
conda env create --file conda.yaml
```
2. Run command:
```
mlflow models serve -m mlruns/0/8ad24d33836340df9a724d04ef972154/artificats/model -p 1234
```
3. Activate conda environment:
```
conda activate RandomForest
```
3. Run the file 
```
sh make_predictions.sh
```