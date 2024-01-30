from label_studio_converter import Converter
import json
import os

BASE_DIR = os.path.dirname(__file__)
TEST_DATA_PATH = os.path.join(BASE_DIR, "data", "test_export_json_min")
INPUT_JSON_PATH = os.path.join(BASE_DIR, TEST_DATA_PATH, "data.json")
LABEL_CONFIG_PATH = os.path.join(BASE_DIR, TEST_DATA_PATH, "label_config.xml")
INPUT_JSON_PATH_REPEATER = os.path.join(BASE_DIR, TEST_DATA_PATH, "data_repeater.json")
LABEL_CONFIG_JSON_PATH_REPEATER = os.path.join(
    BASE_DIR, TEST_DATA_PATH, "label_config_repeater.json"
)

def test_simple_json_min():
    converter = Converter(LABEL_CONFIG_PATH, '/tmp')
    output_dir = '/tmp/lsc-pytest'
    result_json = output_dir + '/result.json'
    input_data = INPUT_JSON_PATH
    converter.convert_to_json_min(input_data, output_dir, is_dir=False)

    loaded_json_min = json.load(open(result_json, 'r'))

    assert len(loaded_json_min) == 1
    assert 'label' in loaded_json_min[0]

def test_repeater_json_min():
    # The config parser built into LSC doesn't recognize regexes in the label config
    # so we used a previously parsed JSON config instead
    json_config = json.load(open(LABEL_CONFIG_JSON_PATH_REPEATER, 'r'))
    converter = Converter(json_config, '/tmp')
    output_dir = '/tmp/lsc-pytest'
    result_json = output_dir + '/result.json'
    input_data = INPUT_JSON_PATH_REPEATER
    converter.convert_to_json_min(input_data, output_dir, is_dir=False)

    loaded_json_min = json.load(open(result_json, 'r'))

    assert len(loaded_json_min) == 1
    assert 'labels_0' in loaded_json_min[0]
    assert 'categories_0' in loaded_json_min[0]
