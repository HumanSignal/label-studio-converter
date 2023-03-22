import json
import os
import glob
import lxml.etree as ET

from label_studio_converter.imports import yolo as import_yolo



def test_base_import_yolo():
    """Tests generated config and json files for yolo imports
    test_import_yolo_data folder assumes only images in the 'images' folder
    with corresponding labels existing in the 'labes' dir and a 'classes.txt' present.
    (currently 7 images -> 3 png, 2 jpg and 2 jpeg files)
    """
    input_data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),'data','test_import_yolo_data')
    out_json_file = os.path.join('/tmp','lsc-pytest','yolo_exp_test.json')

    image_ext = '.jpg,.jpeg,.png' #comma seperated string of extns.

    import_yolo.convert_yolo_to_ls(
        input_dir=input_data_dir,
        out_file=out_json_file,
        image_ext=image_ext
    )

    #'yolo_exp_test.label_config.xml' and 'yolo_exp_test.json' must be generated.
    out_config_file = os.path.join('/tmp','lsc-pytest','yolo_exp_test.label_config.xml')
    assert os.path.exists(out_config_file) and os.path.exists(out_json_file), "> import failed! files not generated."

    #provided labels from classes.txt
    with open(os.path.join(input_data_dir,'classes.txt'), 'r') as f:
        labels = f.read()[:-1].split('\n') #[:-1] since last line in classes.txt is empty by convention

    #generated labels from config xml
    label_element = ET.parse(out_config_file).getroot()[2]
    lables_generated = [x.attrib['value'] for x in label_element.getchildren()]
    assert set(labels) == set(lables_generated), "> generated class labels do not match original labels"

    #total image files in the input folder
    img_files = glob.glob(os.path.join(input_data_dir,'images','*'))

    with open(out_json_file, 'r') as f:
        ls_data = json.loads(f.read())

    assert len(ls_data) == len(img_files), "some file imports did not succeed!"


def test_base_import_yolo_with_img_dims():
    """Tests generated config and json files for yolo imports while importing unique image dims
    """
    input_data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),'data','test_import_yolo_data_unif_dims')
    out_json_file = os.path.join('/tmp','lsc-pytest','yolo_exp_test.json')

    image_ext = '.jpg,.jpeg,.png' # comma seperated string of extns.
    img_dims = (640, 329)  # known width and height of dataset

    import_yolo.convert_yolo_to_ls(
        input_dir=input_data_dir,
        out_file=out_json_file,
        image_ext=image_ext
    )

    # 'yolo_exp_test.label_config.xml' and 'yolo_exp_test.json' must be generated.
    out_config_file = os.path.join('/tmp','lsc-pytest','yolo_exp_test.label_config.xml')
    assert os.path.exists(out_config_file) and os.path.exists(out_json_file), "> import failed! files not generated."

    # provided labels from classes.txt
    with open(os.path.join(input_data_dir, 'classes.txt'), 'r') as f:
        labels = f.read()[:-1].split(
            '\n'
        )  # [:-1] since last line in classes.txt is empty by convention

    # generated labels from config xml
    label_element = ET.parse(out_config_file).getroot()[2]
    lables_generated = [x.attrib['value'] for x in label_element.getchildren()]
    assert set(labels) == set(
        lables_generated
    ), "> generated class labels do not match original labels"

    #total image files in the input folder
    img_files = glob.glob(os.path.join(input_data_dir,'images','*'))

    with open(out_json_file, 'r') as f:
        ls_data = json.loads(f.read())

    assert len(ls_data) == len(img_files), "some file imports did not succeed!"


if __name__ == '__main__':
    test_base_import_yolo()
    test_base_import_yolo_with_img_dims()
