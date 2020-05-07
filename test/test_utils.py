import unittest
import label_studio_converter.utils as utils


class TestUtils(unittest.TestCase):
    def test_tokenize(self):
        # parses string correctly
        val = "parse this string"
        out = utils.tokenize(val)
        assert out == [
            ("parse", 0), 
            ("this", 6) , 
            ("string", 11)]


    # def test_create_tokens_and_tags(self):
    #     s = 'hello world'
    #     self.assertEqual(s.split(), ['hello', 'world'])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)