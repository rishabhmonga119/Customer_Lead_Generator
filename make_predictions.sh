echo "Predicting classes"

#Read data values for batch scoring from batch_values.txt
data=`cat data/batch_values.txt`

#Pass the data to the running prediction server, get the predictions and save them to batch_score_predictions.txt
curl -X POST -H "Content-Type:application/json; format=pandas-split" --data "{\"columns\":[\"param1\", \"param2\", \"param3\", \"param4\", \"param5\", \"param6\", \"param7\", \"param8\", \"param9\", \"param10\", \"param11\", \"param12\", \"param13\", \"param14\", \"param15\", \"param16\", \"param17\", \"param18\", \"param19\", \"param20\", \"param21\", \"param22\", \"param23\", \"param24\"],\"data\":["$data"]}" http://127.0.0.1:1234/invocations > data/batch_score_predictions.txt

echo ""
echo "Done!"

#exec $SHELL