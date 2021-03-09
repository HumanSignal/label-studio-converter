"""
Test for the brush.py module
"""

from label_studio_converter.brush import encode_rle
import unittest


def test_rle_encoding():
    """testing rle encoding with simple example"""
    test_arr = [1, 1, 1, 1, 2, 3, 5, 6, 7, 8, 4, 4, 4, 4, 4, 4, 4, 4]
    rle_test = encode_rle(test_arr)
    assert rle_test == [0, 0, 0, 18, 57, 27, 252, 96, 32, 1, 0, 6, 0, 40, 0,
                        192, 3, 128, 17, 56, 32]
