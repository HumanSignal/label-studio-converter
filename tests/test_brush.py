"""
Test for the brush.py module
"""
import pytest
import unittest
import urllib3
import json
import numpy as np

from label_studio_converter.brush import (
    encode_rle,
    image2annotation,
    binary_mask_to_rle,
    ls_rle_to_coco_rle,
)


def test_image2annotation():
    """
    Import from png to LS annotation with RLE values
    """
    annotation = image2annotation(
        'tests/test.png',
        label_name='Airplane',
        from_name='tag',
        to_name='image',
        model_version='v1',
        score=0.5,
    )

    # prepare Label Studio Task
    task = {
        'data': {'image': 'https://labelstud.io/images/test.jpg'},
        'predictions': [annotation],
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
    """
    Encode from color values of pixels to RLE, simple example
    """
    test_arr = [1, 1, 1, 1, 2, 3, 5, 6, 7, 8, 4, 4, 4, 4, 4, 4, 4, 4]  # color pixels in rgb format
    rle_test = encode_rle(test_arr)  # rle encoded output that will be stored in LS annotations
    assert rle_test == [
        0,
        0,
        0,
        18,
        57,
        27,
        252,
        96,
        32,
        1,
        0,
        6,
        0,
        40,
        0,
        192,
        3,
        128,
        17,
        56,
        32,
    ]


def test_binary_mask_to_rle():
    # Test with a simple binary mask
    binary_mask = np.array([[0, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 1]])
    rle = binary_mask_to_rle(binary_mask)
    assert rle == {'counts': [1, 1, 1, 2, 1, 2, 3, 1], 'size': [3, 4]}

    # Test with a binary mask that is all zeros
    binary_mask = np.zeros((3, 4))
    rle = binary_mask_to_rle(binary_mask)
    assert rle == {'counts': [12], 'size': [3, 4]}

    # Test with a binary mask that is all ones
    binary_mask = np.ones((4, 5))
    rle = binary_mask_to_rle(binary_mask)
    assert rle == {'counts': [0, 20], 'size': [4, 5]}


def test_ls_rle_to_coco_rle():
    # Test with a simple LS RLE
    pytest.importorskip("pycocotools")
    ls_rle = encode_rle([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
    height = 2
    width = 3
    coco_rle = ls_rle_to_coco_rle(ls_rle, height, width)
    assert coco_rle == {'counts': '06', 'size': [2, 3]}

