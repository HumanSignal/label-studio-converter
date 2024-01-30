import label_studio_converter.utils as utils


def test_create_tokens_and_tags_with_eol_tag():
    s = 'I need a break\nplease'
    spans = [
        {
            'end': 14,
            'labels': ['Person'],
            'start': 9,
            'text': 'break',
            'type': 'Labels',
        }
    ]
    tokens, tags = utils.create_tokens_and_tags(s, spans)
    assert tokens[0] == "I"
    assert tags[0] == "O"
    assert tokens[1] == "need"
    assert tags[1] == "O"
    assert tokens[2] == "a"
    assert tags[2] == "O"
    assert tokens[3] == "break"
    assert tags[3] == "B-Person"
    assert tokens[4] == "please"
    assert tags[4] == "O"

def test_create_tokens_and_tags_with_tab_tag():
    s = 'I need a tab\tplease'
    spans = [
        {
            'end': 12,
            'labels': ['Person'],
            'start': 9,
            'text': 'tab',
            'type': 'Labels',
        }
    ]
    tokens, tags = utils.create_tokens_and_tags(s, spans)
    assert tokens[0] == "I"
    assert tags[0] == "O"
    assert tokens[1] == "need"
    assert tags[1] == "O"
    assert tokens[2] == "a"
    assert tags[2] == "O"
    assert tokens[3] == "tab"
    assert tags[3] == "B-Person"
    assert tokens[4] == "please"
    assert tags[4] == "O"

def test_create_tokens_and_tags_for_full_token():
    text = 'We gave Jane Smith the ball.'
    spans = [{'end': 12, 'labels': ['Person'], 'start': 9, 'text': 'ane'}]
    tokens, tags = utils.create_tokens_and_tags(text, spans)
    assert tokens == ['We', 'gave', 'Jane', 'Smith', 'the', 'ball', '.']
    assert tags == ['O', 'O', 'B-Person', 'O', 'O', 'O', 'O']

def test_create_tokens_and_tags_with_leading_space():
    text = 'We gave Jane Smith the ball.'
    spans = [{'end': 18, 'labels': ['Person'], 'start': 7, 'text': ' Jane Smith'}]
    tokens, tags = utils.create_tokens_and_tags(text, spans)
    assert tokens == ['We', 'gave', 'Jane', 'Smith', 'the', 'ball', '.']
    assert tags == ['O', 'O', 'B-Person', 'I-Person', 'O', 'O', 'O']

def test_create_tokens_and_tags_with_trailing_space():
    text = 'We gave Jane Smith the ball.'
    spans = [{'end': 19, 'labels': ['Person'], 'start': 8, 'text': 'Jane Smith '}]
    tokens, tags = utils.create_tokens_and_tags(text, spans)
    assert tokens == ['We', 'gave', 'Jane', 'Smith', 'the', 'ball', '.']
    assert tags == ['O', 'O', 'B-Person', 'I-Person', 'O', 'O', 'O']

def test_create_tokens_and_tags_for_token_with_dollar_sign():
    text = 'A meal with $3.99 bill'
    spans = [{'end': 16, 'labels': ['Person'], 'start': 12, 'text': '$3.99'}]
    tokens, tags = utils.create_tokens_and_tags(text, spans)
    assert tokens == ['A', 'meal', 'with', '$', '3.99', 'bill']
    assert tags == ['O', 'O', 'O', 'B-Person', 'I-Person', 'O']

def test_create_tokens_and_tags_for_token_with_slash_sign():
    text = 'A meal from Google/Facebook'
    spans = [
        {'end': 17, 'labels': ['ORG'], 'start': 12, 'text': 'Google'},
        {'end': 22, 'labels': ['ORG'], 'start': 19, 'text': 'Face'},
    ]
    tokens, tags = utils.create_tokens_and_tags(text, spans)
    assert tokens == ['A', 'meal', 'from', 'Google', '/', 'Facebook']
    assert tags == ['O', 'O', 'O', 'B-ORG', 'O', 'B-ORG']

def test_create_tokens_and_tags_for_token_with_ampersand_sign():
    text = "They're the world's most fear;some fighting team,Teenage Mut;ant Ninja Turtles (We're really hip!).They're heroes in the half-shell and they're green,Teenage Mutant Ninja Turtles (Hey, get a grip!),When the evil Shre&dder attacks These Turtle boys don't cut him no slack! Spli\/nter taught them to be ninja teens.. Teen#age Mutant Ninja Turtles (He's a radi@cal rat!), Leonardo leads, Donat#ello does machines, Teenage Mutant Ninja Turt;les (That's a fact, Jack!),Raphael is cool but rude (Gimme a break!), Michel;angelo is a party dude (Party!)"
    spans = [
        {"start": 0, "end": 4, "text": "They", "labels": ["PER"]},
        {"start": 12, "end": 17, "text": "world", "labels": ["PER"]},
        {"start": 57, "end": 60, "text": "Mut", "labels": ["PER"]},
        {"start": 212, "end": 216, "text": "Shre", "labels": ["PER"]},
        {"start": 314, "end": 318, "text": "Teen", "labels": ["PER"]},
        {"start": 272, "end": 276, "text": "Spli", "labels": ["PER"]},
    ]
    tokens, tags = utils.create_tokens_and_tags(text, spans)
    assert tokens[52] == 'Shre'
    assert tags[52] == 'B-PER'
    assert tags[51] == 'O'
    assert tags[53] == 'O'

def test_create_tokens_and_tags_for_token_with_hash_sign():
    text = "Mut;ant Ninja Turtles. Teen#age Mutant Ninja Turtles."
    spans = [
        {"start": 0, "end": 3, "text": "Mut", "labels": ["PER"]},
        {"start": 5, "end": 7, "text": "ant", "labels": ["ORG"]},
        {"start": 24, "end": 27, "text": "Teen", "labels": ["PER"]},
        {"start": 29, "end": 30, "text": "Shre", "labels": ["ORG"]},
    ]
    tokens, tags = utils.create_tokens_and_tags(text, spans)
    assert tokens == [
        'Mut',
        ';',
        'ant',
        'Ninja',
        'Turtles.',
        'Teen',
        '#',
        'age',
        'Mutant',
        'Ninja',
        'Turtles',
        '.',
    ]

    assert tags == [
        'B-PER',
        'O',
        'B-ORG',
        'O',
        'O',
        'B-PER',
        'O',
        'B-ORG',
        'O',
        'O',
        'O',
        'O',
    ]
