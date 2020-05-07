import unittest
import label_studio_converter.utils as utils


class TestUtils(unittest.TestCase):
    def test_tokenize(self):
        # parses string correctly
        val = "parse this string"
        out = utils.tokenize(val)
        assert out == [
            ("parse", 0),
            ("this", 6),
            ("string", 11)]

    def test_create_tokens_and_tags(self):
        # handles situation when span end exactly equals token end

        s = 'my world Hello not'
        spans = [{'end': 8,
                  'labels': ['GPE'],
                  'start': 0,
                  'text': 'my world',
                  'type': 'Labels'},
                 {'end': 13,
                  'labels': ['PERSON'],
                  'start': 9,
                  'text': 'Hello',
                  'type': 'Labels'}]
        tokens, tags = utils.create_tokens_and_tags(s, spans)
        self.assertEqual(tokens[0], "my")
        self.assertEqual(tags[0], "B-GPE")
        self.assertEqual(tokens[1], "world")
        self.assertEqual(tags[1], "I-GPE")
        self.assertEqual(tokens[2], "Hello")
        self.assertEqual(tags[2], "B-PERSON")
        self.assertEqual(tokens[3], "not")
        self.assertEqual(tags[3], "O")


# for debugging in vscode
if __name__ == '__main__':
    unittest.main()
