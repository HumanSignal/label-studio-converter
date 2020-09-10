import unittest
import label_studio_converter.utils as utils


class TestUtils(unittest.TestCase):
    def test_create_tokens_and_tags_with_eol_tag(self):

        s = 'I need a break\nplease'
        spans = [{'end': 14,
                  'labels': ['Person'],
                  'start': 9,
                  'text': 'break',
                  'type': 'Labels'}]
        tokens, tags = utils.create_tokens_and_tags(s, spans)
        self.assertEqual(tokens[0], "I")
        self.assertEqual(tags[0], "O")
        self.assertEqual(tokens[1], "need")
        self.assertEqual(tags[1], "O")
        self.assertEqual(tokens[2], "a")
        self.assertEqual(tags[2], "O")
        self.assertEqual(tokens[3], "break")
        self.assertEqual(tags[3], "B-Person")
        self.assertEqual(tokens[4], "please")
        self.assertEqual(tags[4], "O")

    def test_create_tokens_and_tags_with_tab_tag(self):

        s = 'I need a tab\tplease'
        spans = [{'end': 12,
                  'labels': ['Person'],
                  'start': 9,
                  'text': 'tab',
                  'type': 'Labels'}]
        tokens, tags = utils.create_tokens_and_tags(s, spans)
        self.assertEqual(tokens[0], "I")
        self.assertEqual(tags[0], "O")
        self.assertEqual(tokens[1], "need")
        self.assertEqual(tags[1], "O")
        self.assertEqual(tokens[2], "a")
        self.assertEqual(tags[2], "O")
        self.assertEqual(tokens[3], "tab")
        self.assertEqual(tags[3], "B-Person")
        self.assertEqual(tokens[4], "please")
        self.assertEqual(tags[4], "O")
