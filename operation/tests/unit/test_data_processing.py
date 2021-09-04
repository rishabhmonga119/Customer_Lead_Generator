import unittest
import pandas as pd
import sys
sys.path.insert(1, '../src')
import data_processing


class TestData(unittest.TestCase):
    def test_cleaned_data(self):
        df = pd.DataFrame({'col1': [1, 2, '3', 4], 'col2': ["", '4', 5, '6.0'], 'col3': [1, '3', 5, 'err']})
        df_cleaned = data_processing.data_cleaning(df)
        self.assertEqual(len(df_cleaned),2)