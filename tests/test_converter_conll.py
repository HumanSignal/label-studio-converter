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

    def test_create_tokens_and_tags_for_full_token(self):
        text = 'We gave Jane Smith the ball.'
        spans = [{'end': 12, 'labels': ['Person'], 'start': 9, 'text': 'ane'}]
        tokens, tags = utils.create_tokens_and_tags(text, spans)
        self.assertEqual(tokens, ['We', 'gave', 'Jane', 'Smith', 'the', 'ball', '.'])
        self.assertEqual(tags, ['O', 'O', 'B-Person', 'O', 'O', 'O', 'O'])

    def test_create_tokens_and_tags_with_leading_space(self):
        text = 'We gave Jane Smith the ball.'
        spans = [{'end': 18, 'labels': ['Person'], 'start': 7, 'text': ' Jane Smith'}]
        tokens, tags = utils.create_tokens_and_tags(text, spans)
        self.assertEqual(tokens, ['We', 'gave', 'Jane', 'Smith', 'the', 'ball', '.'])
        self.assertEqual(tags, ['O', 'O', 'B-Person', 'I-Person', 'O', 'O', 'O'])

    def test_create_tokens_and_tags_with_trailing_space(self):
        text = 'We gave Jane Smith the ball.'
        spans = [{'end': 19, 'labels': ['Person'], 'start': 8, 'text': 'Jane Smith '}]
        tokens, tags = utils.create_tokens_and_tags(text, spans)
        self.assertEqual(tokens, ['We', 'gave', 'Jane', 'Smith', 'the', 'ball', '.'])
        self.assertEqual(tags, ['O', 'O', 'B-Person', 'I-Person', 'O', 'O', 'O'])

    def test_create_tokens_and_tags_for_token_with_dollar_sign(self):
        text = 'A meal with $3.99 bill'
        spans = [{'end': 16, 'labels': ['Person'], 'start': 12, 'text': '$3.99'}]
        tokens, tags = utils.create_tokens_and_tags(text, spans)
        self.assertEqual(tokens, ['A', 'meal', 'with', '$', '3.99', 'bill'])
        self.assertEqual(tags, ['O', 'O', 'O', 'B-Person', 'I-Person', 'O'])

    def test_create_tokens_and_tags_for_token_with_slash_sign(self):
        text = 'A meal from Google/Facebook'
        spans = [{'end': 17, 'labels': ['ORG'], 'start': 12, 'text': 'Google'},
                 {'end': 22, 'labels': ['ORG'], 'start': 19, 'text': 'Face'}]
        tokens, tags = utils.create_tokens_and_tags(text, spans)
        self.assertEqual(tokens, ['A', 'meal', 'from', 'Google', '/', 'Facebook'])
        self.assertEqual(tags, ['O', 'O', 'O', 'B-ORG', 'O', 'B-ORG'])
