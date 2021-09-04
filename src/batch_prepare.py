import pandas as pd

def prepare_batch(data,BATCH_SIZE, filename):
    """
    prepares the batch dataset and writes this dataset in a file

    :param data: batch data to be prepared
    :type:Pandas DataFrame

    :param BATCH_SIZE: Size of batch for making predictions
    :rtype:Int

    :param filename: Complete file name of the file into which the prepared data is to be written
    :rtype:String
    """
    ### select the last BATCH_SIZE rows from batch dataset
    batch = data.iloc[-BATCH_SIZE:].values.tolist()
    batch_data = []
    
    ### remove white spaces in the list because predtion server expects no white spaces between elements
    for i in batch:
        str_row = str(i)
        str_row = str_row.replace(' ','')
        batch_data.append(str_row)
    
    ### write values in a file called filename
    with open(filename, 'w') as f:
        f.write(','.join(str(i) for i in batch_data))


def main():
    try:
        batch_data = pd.read_csv('data/batch_data.csv', header=None)
        BATCH_SIZE = len(batch_data)
        prepare_batch(batch_data, BATCH_SIZE, 'data/batch_values.txt')
    except:
        return


if __name__ == "__main__":
    main()