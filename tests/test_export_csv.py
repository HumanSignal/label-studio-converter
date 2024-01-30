import json
import os


from label_studio_converter import Converter
from pandas import read_csv


def test_simple_csv_export():
    # Test case 1, simple output, no JSON
    converter = Converter({}, '/tmp')
    output_dir = '/tmp/lsc-pytest'
    result_csv = output_dir + '/result.csv'
    input_data = os.path.abspath(os.path.dirname(__file__)) + '/data/test_export_csv/csv_test.json'
    sep = ','
    converter.convert_to_csv(input_data, output_dir, sep=sep, header=True, is_dir=False)

    df = read_csv(result_csv, sep=sep)
    nulls = df.isnull().sum()
    if nulls.any() > 0:
        assert False, "There should be no empty values in result CSV"


def test_csv_export_complex_fields_with_json():
    converter = Converter({}, '/tmp')
    output_dir = '/tmp/lsc-pytest'
    result_csv = output_dir + '/result.csv'
    input_data = os.path.abspath(os.path.dirname(__file__)) + '/data/test_export_csv/csv_test2.json'
    assert_csv = os.path.abspath(os.path.dirname(__file__)) + '/data/test_export_csv/csv_test2_result.csv'
    sep = '\t'
    converter.convert_to_csv(input_data, output_dir, sep=sep, header=True, is_dir=False)
    df = read_csv(result_csv, sep=sep)
    nulls = df.isnull().sum()
    assert sum(nulls) == 2, "There should be exactly two empty values in result CSV"

    # Ensure fields are valid JSON
    json.loads(df.iloc[0].writers)
    json.loads(df.iloc[0].iswcs_1)

    assert open(result_csv).read() == open(assert_csv).read()
