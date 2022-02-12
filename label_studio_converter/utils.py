import io
import os
import xml.etree.ElementTree
import requests
import hashlib
import logging
import urllib
import numpy as np
import wave
import shutil
import argparse

from operator import itemgetter
from PIL import Image
from urllib.parse import urlparse
from nltk.tokenize import WhitespaceTokenizer
from lxml import etree
from collections import defaultdict

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
    if spans and all([span.get('start') is not None and span.get('end') is not None for span in spans]):
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


def download(url, output_dir, filename=None, project_dir=None, return_relative_path=False, upload_dir=None,
             download_resources=True):
    is_local_file = url.startswith('/data/') and '?d=' in url
    is_uploaded_file = url.startswith('/data/upload')

    if is_uploaded_file:
        upload_dir = _get_upload_dir(project_dir, upload_dir)
        filename = url.replace('/data/upload/', '')
        filepath = os.path.join(upload_dir, filename)
        logger.debug(f'Copy {filepath} to {output_dir}'.format(filepath=filepath, output_dir=output_dir))
        if download_resources:
            shutil.copy(filepath, output_dir)
        if return_relative_path:
            return os.path.join(os.path.basename(output_dir), filename)
        return filepath

    if is_local_file:
        filename, dir_path = url.split('/data/', 1)[-1].split('?d=')
        dir_path = str(urllib.parse.unquote(dir_path))
        if not os.path.exists(dir_path):
            raise FileNotFoundError(dir_path)
        filepath = os.path.join(dir_path, filename)
        if return_relative_path:
            raise NotImplementedError()
        return filepath

    if filename is None:
        basename, ext = os.path.splitext(os.path.basename(urlparse(url).path))
        filename = basename + '_' + hashlib.md5(url.encode()).hexdigest()[:4] + ext
    filepath = os.path.join(output_dir, filename)
    if not os.path.exists(filepath):
        logger.info('Download {url} to {filepath}'.format(url=url, filepath=filepath))
        if download_resources:
            r = requests.get(url)
            r.raise_for_status()
            with io.open(filepath, mode='wb') as fout:
                fout.write(r.content)
    if return_relative_path:
        return os.path.join(os.path.basename(output_dir), filename)
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
        :param config_string: Label config string
        :return: structured config of the form:
        {
            "<ControlTag>.name": {
                "type": "ControlTag",
                "to_name": ["<ObjectTag1>.name", "<ObjectTag2>.name"],
                "inputs: [
                    {"type": "ObjectTag1", "value": "<ObjectTag1>.value"},
                    {"type": "ObjectTag2", "value": "<ObjectTag2>.value"}
                ],
                "labels": ["Label1", "Label2", "Label3"] // taken from "alias" if exists or "value"
        }
        """
    if not config_string:
        return {}

    def _is_input_tag(tag):
        return tag.attrib.get('name') and tag.attrib.get('value')

    def _is_output_tag(tag):
        return tag.attrib.get('name') and tag.attrib.get('toName') and tag.tag not in _NOT_CONTROL_TAGS

    def _get_parent_output_tag_name(tag, outputs):
        # Find parental <Choices> tag for nested tags like <Choices><View><View><Choice>...
        parent = tag
        while True:
            parent = parent.getparent()
            if parent is None:
                return
            name = parent.attrib.get('name')
            if name in outputs:
                return name

    try:
        xml_tree = etree.fromstring(config_string)
    except etree.XMLSyntaxError as e:
        raise ValueError(str(e))

    inputs, outputs, labels = {}, {}, defaultdict(dict)
    for tag in xml_tree.iter():
        if _is_output_tag(tag):
            tag_info = {'type': tag.tag, 'to_name': tag.attrib['toName'].split(',')}
            # Grab conditionals if any
            conditionals = {}
            if tag.attrib.get('perRegion') == 'true':
                if tag.attrib.get('whenTagName'):
                    conditionals = {'type': 'tag', 'name': tag.attrib['whenTagName']}
                elif tag.attrib.get('whenLabelValue'):
                    conditionals = {'type': 'label', 'name': tag.attrib['whenLabelValue']}
                elif tag.attrib.get('whenChoiceValue'):
                    conditionals = {'type': 'choice', 'name': tag.attrib['whenChoiceValue']}
            if conditionals:
                tag_info['conditionals'] = conditionals
            outputs[tag.attrib['name']] = tag_info
        elif _is_input_tag(tag):
            inputs[tag.attrib['name']] = {'type': tag.tag, 'value': tag.attrib['value'].lstrip('$')}
        if tag.tag not in _LABEL_TAGS:
            continue
        parent_name = _get_parent_output_tag_name(tag, outputs)
        if parent_name is not None:
            actual_value = tag.attrib.get('alias') or tag.attrib.get('value')
            if not actual_value:
                logger.debug(
                    'Inspecting tag {tag_name}... found no "value" or "alias" attributes.'.format(
                        tag_name=etree.tostring(tag, encoding='unicode').strip()[:50]))
            else:
                labels[parent_name][actual_value] = dict(tag.attrib)
    for output_tag, tag_info in outputs.items():
        tag_info['inputs'] = []
        for input_tag_name in tag_info['to_name']:
            if input_tag_name not in inputs:
                logger.warning(
                    f'to_name={input_tag_name} is specified for output tag name={output_tag}, '
                    'but we can\'t find it among input tags')
                continue
            tag_info['inputs'].append(inputs[input_tag_name])
        tag_info['labels'] = list(labels[output_tag])
        tag_info['labels_attrs'] = labels[output_tag]
    return outputs


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
