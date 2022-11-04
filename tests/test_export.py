import json
import os

from label_studio_converter import Converter
from pandas import read_csv


def test_csv_export():
    # Test case 1, simple output, no JSON
    converter = Converter({}, '/tmp')
    output_dir = '/tmp/lsc-pytest'
    result_csv = output_dir + '/result.csv'
    input_data = os.path.abspath(os.path.dirname(__file__)) + '/data/csv_test.json'
    sep = ','
    converter.convert_to_csv(input_data, output_dir, sep=sep, header=True, is_dir=False)

    df = read_csv(result_csv, sep=sep) 
    nulls = df.isnull().sum()
    if nulls.any() > 0:
        assert False, "There should be no empty values in result CSV"

    # Test case 2, complex fields with JSON
    input_data = os.path.abspath(os.path.dirname(__file__)) + '/data/csv_test2.json'
    sep = '\t'
    converter.convert_to_csv(input_data, output_dir, sep=sep, header=True, is_dir=False)
    df = read_csv(result_csv, sep=sep) 
    nulls = df.isnull().sum()
    assert sum(nulls) == 2, "There should be exactly two empty values in result CSV"

    # Ensure fields are valid JSON
    json.loads(df.iloc[0].writers)
    json.loads(df.iloc[0].iswcs_1)

if __name__ == '__main__':
    test_csv_export()
