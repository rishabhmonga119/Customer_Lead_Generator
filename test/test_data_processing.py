import unittest
import pandas as pd
import sys
sys.path.insert(1, 'src')
print(sys.path)
import data_processing


class TestData(unittest.TestCase):
    
    def test_cleaned_data(self):
        """
        This test is used to ensure all rows with missing values and consisting of strings not convertable to float are dropped.
        """
        ### dataframe created with one row with missing value and one row with non-float-convertible string. Successful removal of the two rows
        ### will result in a dataframe of length two. 
        df = pd.DataFrame({'col1': [1, 2, '3', 4], 'col2': ["", '4', 5, '6.0'], 'col3': [1, '3', 5, 'err']})
        df_cleaned = data_processing.data_cleaning(df)
        self.assertEqual(len(df_cleaned),2)

if __name__ == '__main__':
    unittest.main()