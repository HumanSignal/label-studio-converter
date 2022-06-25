"""
Test for the brush.py module
"""
import unittest
import urllib3
import json

from label_studio_converter.brush import encode_rle, image2annotation


def test_image2annotation():
    annotation = image2annotation(
        'tests/test.png',
        label_name='Airplane', from_name='tag', to_name='image',
        model_version='v1', score=0.5
    )

    # prepare Label Studio Task
    task = {
        'data': {'image': 'https://labelstud.io/images/test.jpg'},
        'predictions': [
            annotation
        ]
    }

    """ You can import this `task.json` to the Label Studio project with this labeling config:
    
    <View>
      <Image name="image" value="$image" zoom="true"/>
      <BrushLabels name="tag" toName="image">
        <Label value="Airplane" background="rgba(255, 0, 0, 0.7)"/>
        <Label value="Car" background="rgba(0, 0, 255, 0.7)"/>
      </BrushLabels>
    </View>

    """
    json.dump(task, open('task.json', 'w'))
    assert True


def test_rle_encoding():
    """testing rle encoding with simple example"""
    test_arr = [1, 1, 1, 1, 2, 3, 5, 6, 7, 8, 4, 4, 4, 4, 4, 4, 4, 4]
    rle_test = encode_rle(test_arr)
    assert rle_test == [0, 0, 0, 18, 57, 27, 252, 96, 32, 1, 0, 6, 0, 40, 0,
                        192, 3, 128, 17, 56, 32]
