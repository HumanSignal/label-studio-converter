import unittest
from label_studio_converter.converter import Converter
import os
import json
CONLL_OUT_PATH = "tmp/completions/result.conll"


class TestConverter(unittest.TestCase):
    def setUp(self):
        os.remove(CONLL_OUT_PATH)
        assert not os.path.isfile(CONLL_OUT_PATH)

    def test_convert_conll(self):
        c = Converter("test/fixtures/config.xml")
        c.convert_to_conll2003('test/fixtures/completions/', 'tmp/completions')
        self.assertTrue(os.path.isfile(CONLL_OUT_PATH))

        with open(CONLL_OUT_PATH, 'r') as file:
            data = file.read()
        with open("test/fixtures/completions/348.json", 'r') as file:
            completion = json.loads(file.read())

        results = completion["completions"][0]["result"]
        for result in results:
            substrings = result["value"]["text"].split(" ")
            substrings = [r'{}'.format(s.replace("(", "\(")
                                       .replace(")", "\)")
                                       .replace(".", "\.")
                                       .replace(",", "\,"))

                          for s in substrings if s]
            # first substring should tagged as beginning
            first_token = substrings[0]
            if len(substrings) > 1:
                second_token = substrings[1]
            else:
                second_token = None
            with self.subTest(first_token=first_token, second_token=second_token):
                self.assertRegex(data, rf"{first_token}[\,\.]* -X- _ B-.+")
                if second_token:
                    self.assertRegex(
                        data, rf"{second_token}[\,\.]* -X- _ I-.+")


# for debugging in vscode
if __name__ == '__main__':
    unittest.main()
