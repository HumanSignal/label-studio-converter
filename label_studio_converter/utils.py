import io
import os
import warnings

import requests
import hashlib
import logging
import urllib
import numpy as np
import wave
import shutil
import argparse

from pathlib import Path
from operator import itemgetter
from PIL import Image
from nltk.tokenize import WhitespaceTokenizer
import label_studio_tools.core.label_config as label_config
from label_studio_tools.core.utils.io import get_local_path

logger = logging.getLogger(__name__)

_LABEL_TAGS = {'Label', 'Choice'}
_NOT_CONTROL_TAGS = {'Filter',}


class ExpandFullPath(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))


def tokenize(text):
    tok_start = 0
    out = []
    for tok in text.split():
        if len(tok):
            out.append((tok, tok_start))
            tok_start += len(tok) + 1
        else:
            tok_start += 1
    return out


def create_tokens_and_tags(text, spans):
    #tokens_and_idx = tokenize(text) # This function doesn't work properly if text contains multiple whitespaces...
    token_index_tuples = [token for token in WhitespaceTokenizer().span_tokenize(text)]
    tokens_and_idx = [(text[start:end], start) for start, end in token_index_tuples]
    spans = [span for span in spans if span.get('type') == 'labels']
    if spans:
        spans = list(sorted(spans, key=itemgetter('start')))
        span = spans.pop(0)
        span_start = span['start']
        span_end = span['end']-1
        prefix = 'B-'
        tokens, tags = [], []
        for token, token_start in tokens_and_idx:
            tokens.append(token)
            token_end = token_start + len(token) #"- 1" - This substraction is wrong. token already uses the index E.g. "Hello" is 0-4
            token_start_ind = token_start  #It seems like the token start is too early.. for whichever reason

            #if for some reason end of span is missed.. pop the new span (Which is quite probable due to this method)
            #Attention it seems like span['end'] is the index of first char afterwards. In case the whitespace is part of the
            #labell we need to subtract one. Otherwise next token won't trigger the span update.. only the token after next..
            if token_start_ind > span_end:
                while spans:
                    span = spans.pop(0)
                    span_start = span['start']
                    span_end = span['end'] - 1
                    prefix = 'B-'
                    if token_start <= span_end:
                        break
            # Add tag "O" for spans that:
            # - are empty
            # - span start has passed over token_end
            # - do not have any label (None or empty list)
            if not span or token_end < span_start or not span.get('labels'):
                tags.append('O')
            elif span_start <= token_end and span_end >= token_start_ind:
                tags.append(prefix + span['labels'][0])
                prefix = 'I-'
            else:
                tags.append('O')
    else:
        tokens = [token for token, _ in tokens_and_idx]
        tags = ['O'] * len(tokens)

    return tokens, tags


def _get_upload_dir(project_dir=None, upload_dir=None):
    """Return either upload_dir, or path by LS_UPLOAD_DIR, or project_dir/upload"""
    if upload_dir:
        return upload_dir
    upload_dir = os.environ.get('LS_UPLOAD_DIR')
    if not upload_dir and project_dir:
        upload_dir = os.path.join(project_dir, 'upload')
        if not os.path.exists(upload_dir):
            upload_dir = None
    if not upload_dir:
        raise FileNotFoundError("Can't find upload dir: either LS_UPLOAD_DIR or project should be passed to converter")
    return upload_dir


def download(url,
             output_dir,
             filename=None,
             project_dir=None,
             return_relative_path=False,
             upload_dir=None,
             download_resources=True):
    is_local_file = url.startswith('/data/') and '?d=' in url

    filepath = get_local_path(url=url,
                              cache_dir=output_dir,
                              image_dir=_get_upload_dir(project_dir, upload_dir),
                              access_token="access_token",
                              download_resources=download_resources)
    new_filename = Path(filepath)

    if is_local_file:
        if return_relative_path:
            raise NotImplementedError()
        return filepath
    if return_relative_path:
        return os.path.join(os.path.basename(output_dir), new_filename.name)
    return filepath


def get_image_size(image_path):
    return Image.open(image_path).size


def get_image_size_and_channels(image_path):
    i = Image.open(image_path)
    w, h = i.size
    c = len(i.getbands())
    return w, h, c


def get_audio_duration(audio_path):
    with wave.open(audio_path, mode='r') as f:
        return f.getnframes() / float(f.getframerate())


def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def parse_config(config_string):
    """
    Method label_studio_converter.parse_config is deprecated.
    Please use label_studio_tools.core.label_config.parse_config
    """
    warnings.warn(
        "label_studio_converter.parse_config is deprecated. "
        "Please use label_studio_tools.core.label_config.parse_config", DeprecationWarning
    )
    return label_config.parse_config(config_string)


def get_polygon_area(x, y):
    """https://en.wikipedia.org/wiki/Shoelace_formula"""

    assert len(x) == len(y)

    return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def get_polygon_bounding_box(x, y):
    assert len(x) == len(y)

    x1, y1, x2, y2 = min(x), min(y), max(x), max(y)
    return [x1, y1, x2 - x1, y2 - y1]


def _get_annotator(item, default=None, int_id=False):
    """ Get annotator id or email from annotation
    """
    annotator = item['completed_by']
    if isinstance(annotator, dict):
        annotator = annotator.get('email', default)
        return annotator

    if isinstance(annotator, int) and int_id:
        return annotator

    return str(annotator)
