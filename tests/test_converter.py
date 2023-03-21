from label_studio_converter import Converter
import os
import pytest
import tempfile
import shutil

BASE_DIR = os.path.dirname(__file__)
TEST_DATA_PATH = os.path.join(BASE_DIR, "data", "test_converter_data")
INPUT_JSON_PATH = os.path.join(BASE_DIR, TEST_DATA_PATH, "data.json")
LABEL_CONFIG_PATH = os.path.join(BASE_DIR, TEST_DATA_PATH, "label_config.xml")
INPUT_JSON_PATH_POLYGONS = os.path.join(BASE_DIR, TEST_DATA_PATH, "data_polygons.json")
LABEL_CONFIG_PATH_POLYGONS = os.path.join(
    BASE_DIR, TEST_DATA_PATH, "label_config_polygons.xml"
)


def check_equal_list_of_strings(list1, list2):
    # Check that both lists are not empty
    if not list1 or not list2:
        return False

    list1.sort()
    list2.sort()

    # Check that the lists have the same length
    if len(list1) != len(list2):
        return False

    # Check that the elements of the lists are equal
    for i in range(len(list1)):
        if list1[i] != list2[i]:
            return False

    return True


def get_os_walk(root_path):
    list_file_paths = []
    for root, dirs, files in os.walk(root_path):
        for f in files:
            list_file_paths += [os.path.join(root, f)]
    return list_file_paths


@pytest.fixture
def create_temp_folder():
    # Create a temporary folder
    temp_dir = tempfile.mkdtemp()

    # Yield the temporary folder
    yield temp_dir

    # Remove the temporary folder after the test
    shutil.rmtree(temp_dir)


def test_convert_to_yolo(create_temp_folder):
    """Check converstion label_studio json exported file to yolo with multiple labelers"""

    # Generates a temporary folder and return the absolute path
    # The temporary folder contains all the data generate by the following function
    # For debugging replace create_temp_folder with "./tmp"
    tmp_folder = create_temp_folder

    output_dir = tmp_folder
    output_image_dir = os.path.join(output_dir, "tmp_image")
    output_label_dir = os.path.join(output_dir, "tmp_label")
    project_dir = "."

    converter = Converter(LABEL_CONFIG_PATH, project_dir)
    converter.convert_to_yolo(
        INPUT_JSON_PATH,
        output_dir,
        output_image_dir=output_image_dir,
        output_label_dir=output_label_dir,
        is_dir=False,
        split_labelers=True,
    )

    abs_path_label_dir = os.path.abspath(output_label_dir)
    expected_paths = [
        os.path.join(abs_path_label_dir, "1", "image1.txt"),
        os.path.join(abs_path_label_dir, "1", "image2.txt"),
        os.path.join(abs_path_label_dir, "2", "image1.txt"),
    ]
    generated_paths = get_os_walk(abs_path_label_dir)
    # Check all files and subfolders have been generated.
    assert check_equal_list_of_strings(
        expected_paths, generated_paths
    ), f"Generated file: \n  {generated_paths} \n does not match expected ones: \n {expected_paths}"
    # Check all the annotations have been converted to yolo
    for file in expected_paths:
        with open(file) as f:
            lines = f.readlines()
            assert (
                len(lines) == 2
            ), f"Expect different number of annotations in file {file}."


def test_convert_polygons_to_yolo(create_temp_folder):
    """Check converstion label_studio json exported file to yolo with polygons"""

    # Generates a temporary folder and return the absolute path
    # The temporary folder contains all the data generate by the following function
    # For debugging replace create_temp_folder with "./tmp"
    tmp_folder = create_temp_folder

    output_dir = tmp_folder
    output_image_dir = os.path.join(output_dir, "tmp_image")
    output_label_dir = os.path.join(output_dir, "tmp_label")
    project_dir = "."

    converter = Converter(LABEL_CONFIG_PATH_POLYGONS, project_dir)
    converter.convert_to_yolo(
        INPUT_JSON_PATH_POLYGONS,
        output_dir,
        output_image_dir=output_image_dir,
        output_label_dir=output_label_dir,
        is_dir=False,
        split_labelers=False,
    )

    abs_path_label_dir = os.path.abspath(output_label_dir)
    expected_paths = [os.path.join(abs_path_label_dir, "image2.txt")]
    generated_paths = get_os_walk(abs_path_label_dir)
    # Check all files and subfolders have been generated.
    assert check_equal_list_of_strings(
        expected_paths, generated_paths
    ), f"Generated file: \n  {generated_paths} \n does not match expected ones: \n {expected_paths}"
    # Check all the annotations have been converted to yolo
    for file in expected_paths:
        with open(file) as f:
            lines = f.readlines()
            assert (
                len(lines) == 1
            ), f"Expect different number of annotations in file {file}."
