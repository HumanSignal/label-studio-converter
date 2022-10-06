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


    def test_create_tokens_and_tags_for_token_with_ampersand_sign(self):
        text = "They're the world's most fear;some fighting team,Teenage Mut;ant Ninja Turtles (We're really hip!).They're heroes in the half-shell and they're green,Teenage Mutant Ninja Turtles (Hey, get a grip!),When the evil Shre&dder attacks These Turtle boys don't cut him no slack! Spli\/nter taught them to be ninja teens.. Teen#age Mutant Ninja Turtles (He's a radi@cal rat!), Leonardo leads, Donat#ello does machines, Teenage Mutant Ninja Turt;les (That's a fact, Jack!),Raphael is cool but rude (Gimme a break!), Michel;angelo is a party dude (Party!)"
        spans = [{"start":0,"end":4,"text":"They","labels":["PER"]},{"start":12,"end":17,"text":"world","labels":["PER"]},{"start":57,"end":60,"text":"Mut","labels":["PER"]},{"start":212,"end":216,"text":"Shre","labels":["PER"]},{"start":314,"end":318,"text":"Teen","labels":["PER"]},{"start":272,"end":276,"text":"Spli","labels":["PER"]}]
        tokens, tags = utils.create_tokens_and_tags(text, spans)
        self.assertEqual(tokens[52], 'Shre')
        self.assertEqual(tags[52], 'B-PER')
        self.assertEqual(tags[51], 'O')
        self.assertEqual(tags[53], 'O')


    def test_create_tokens_and_tags_for_token_with_hash_sign(self):
        text = "Mut;ant Ninja Turtles. Teen#age Mutant Ninja Turtles."
        spans = [{"start":0,"end":3,"text":"Mut","labels":["PER"]},
                 {"start":5,"end":7,"text":"ant","labels":["ORG"]},
                 {"start":24,"end":27,"text":"Teen","labels":["PER"]},
                 {"start":29,"end":30,"text":"Shre","labels":["ORG"]}]
        tokens, tags = utils.create_tokens_and_tags(text, spans)
        self.assertEqual(tokens, ['Mut', ';', 'ant', 'Ninja', 'Turtles.', 'Teen', '#', 'age', 'Mutant', 'Ninja', 'Turtles', '.'])
        self.assertEqual(tags, ['B-PER', 'O', 'B-ORG', 'O', 'O', 'B-PER', 'O', 'B-ORG', 'O', 'O', 'O', 'O'])