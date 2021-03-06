echo "Predicting classes"

# Prepare the batch data which stores result in batch_values.txt or batch_values_test.txt
python src/batch_prepare.py

# Read data values for batch scoring from batch_values.txt
data=`cat data/batch_values.txt`
if [ ${#data} -eq 0 ]; then data=`cat data/batch_values_test.txt`; else echo "..."; fi

# Pass the data to the running MLFlow prediction server, get the predictions and save them to batch_score_predictions.txt
curl -X POST -H "Content-Type:application/json; format=pandas-split" --data "{\"columns\":[\"param1\", \"param2\", \"param3\", \"param4\", \"param5\", \"param6\", \"param7\", \"param8\", \"param9\", \"param10\", \"param11\", \"param12\", \"param13\", \"param14\", \"param15\", \"param16\", \"param17\", \"param18\", \"param19\", \"param20\", \"param21\", \"param22\", \"param23\", \"param24\"],\"data\":["$data"]}" http://127.0.0.1:1234/invocations > data/batch_score_predictions.txt

# Evaluate model performance on the batch data 
python src/batch_score.py

echo ""
echo "Done!"

exec $SHELL
