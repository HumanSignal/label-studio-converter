import os
import json
import io
import logging
import pandas as pd
import xml.dom
import xml.dom.minidom

from shutil import copy2
from enum import Enum
from datetime import datetime
from glob import glob
from collections import Mapping, defaultdict
from operator import itemgetter
from copy import deepcopy

from label_studio_converter.utils import (
    parse_config, create_tokens_and_tags, download, get_image_size, get_image_size_and_channels, ensure_dir,
    get_polygon_area, get_polygon_bounding_box
)
from label_studio_converter import brush
from label_studio_converter.audio import convert_to_asr_json_manifest

logger = logging.getLogger(__name__)


class FormatNotSupportedError(NotImplementedError):
    pass


class Format(Enum):
    JSON = 1
    JSON_MIN = 2
    CSV = 3
    TSV = 4
    CONLL2003 = 5
    COCO = 6
    VOC = 7
    BRUSH_TO_NUMPY = 8
    BRUSH_TO_PNG = 9
    ASR_MANIFEST = 10

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return Format[s]
        except KeyError:
            raise ValueError()


class Converter(object):

    _FORMAT_INFO = {
        Format.JSON: {
            'title': 'JSON',
            'description': 'List of items in raw JSON format stored in one JSON file. Use to export both the data and the annotations for a dataset.',
            'link': 'https://labelstud.io/guide/export.html#JSON'
        },
        Format.JSON_MIN: {
            'title': 'JSON-MIN',
            'description': 'List of items where only "from_name", "to_name" values from the raw JSON format are exported. Use to export only the annotations for a dataset.',
            'link': 'https://labelstud.io/guide/export.html#JSON-MIN',
        },
        Format.CSV: {
            'title': 'CSV',
            'description': 'Results are stored as comma-separated values with the column names specified by the values of the "from_name" and "to_name" fields.',
            'link': 'https://labelstud.io/guide/export.html#CSV'
        },
        Format.TSV: {
            'title': 'TSV',
            'description': 'Results are stored in tab-separated tabular file with column names specified by "from_name" "to_name" values',
            'link': 'https://labelstud.io/guide/export.html#TSV'
        },
        Format.CONLL2003: {
            'title': 'CONLL2003',
            'description': 'Popular format used for the CoNLL-2003 named entity recognition challenge.',
            'link': 'https://labelstud.io/guide/export.html#CONLL2003',
            'tags': ['sequence labeling', 'text tagging', 'named entity recognition']
        },
        Format.COCO: {
            'title': 'COCO',
            'description': 'Popular machine learning format used by the COCO dataset for object detection and image segmentation tasks.',
            'link': 'https://labelstud.io/guide/export.html#COCO',
            'tags': ['image segmentation', 'object detection']
        },
        Format.VOC: {
            'title': 'Pascal VOC XML',
            'description': 'Popular XML-formatted task data used for object detection and image segmentation tasks.',
            'link': 'https://labelstud.io/guide/export.html#Pascal-VOC-XML',
            'tags': ['image segmentation', 'object detection']
        },
        Format.BRUSH_TO_NUMPY: {
            'title': 'Brush labels to NumPy',
            'description': 'Export your brush labels as NumPy 2d arrays. Each label outputs as one image.',
            'link': 'https://labelstud.io/guide/export.html#Brush-labels-to-NumPy-amp-PNG',
            'tags': ['image segmentation']
        },
        Format.BRUSH_TO_PNG: {
            'title': 'Brush labels to PNG',
            'description': 'Export your brush labels as PNG images. Each label outputs as one image.',
            'link': 'https://labelstud.io/guide/export.html#Brush-labels-to-NumPy-amp-PNG',
            'tags': ['image segmentation']
        },
        Format.ASR_MANIFEST: {
            'title': 'ASR Manifest',
            'description': 'Export audio transcription labels for automatic speech recognition as the JSON manifest format expected by NVIDIA NeMo models.',
            'link': 'https://labelstud.io/guide/export.html#ASR-MANIFEST',
            'tags': ['speech recognition']
        }
    }

    def all_formats(self):
        return self._FORMAT_INFO

    def __init__(self, config, project_dir, output_tags=None, upload_dir=None):
        self.project_dir = project_dir
        self.upload_dir = upload_dir
        if isinstance(config, dict):
            self._schema = config
        elif isinstance(config, str):
            if os.path.isfile(config):
                with io.open(config) as f:
                    config_string = f.read()
            else:
                config_string = config
            self._schema = parse_config(config_string)

        self._data_keys, self._output_tags = self._get_data_keys_and_output_tags(output_tags)
        self._supported_formats = self._get_supported_formats()

    def convert(self, input_data, output_data, format, is_dir=True, **kwargs):
        if isinstance(format, str):
            format = Format.from_string(format)
        if format == Format.JSON:
            self.convert_to_json(input_data, output_data, is_dir=is_dir)
        elif format == Format.JSON_MIN:
            self.convert_to_json_min(input_data, output_data, is_dir=is_dir)
        elif format == Format.CSV:
            header = kwargs.get('csv_header', True)
            sep = kwargs.get('csv_separator', ',')
            self.convert_to_csv(input_data, output_data, sep=sep, header=header, is_dir=is_dir)
        elif format == Format.TSV:
            header = kwargs.get('csv_header', True)
            sep = kwargs.get('csv_separator', '\t')
            self.convert_to_csv(input_data, output_data, sep=sep, header=header, is_dir=is_dir)
        elif format == Format.CONLL2003:
            self.convert_to_conll2003(input_data, output_data, is_dir=is_dir)
        elif format == Format.COCO:
            image_dir = kwargs.get('image_dir')
            self.convert_to_coco(input_data, output_data, output_image_dir=image_dir, is_dir=is_dir)
        elif format == Format.VOC:
            image_dir = kwargs.get('image_dir')
            self.convert_to_voc(input_data, output_data, output_image_dir=image_dir, is_dir=is_dir)
        elif format == Format.BRUSH_TO_NUMPY:
            items = self.iter_from_dir(input_data) if is_dir else self.iter_from_json_file(input_data)
            brush.convert_task_dir(items, output_data, out_format='numpy')
        elif format == Format.BRUSH_TO_PNG:
            items = self.iter_from_dir(input_data) if is_dir else self.iter_from_json_file(input_data)
            brush.convert_task_dir(items, output_data, out_format='png')
        elif format == Format.ASR_MANIFEST:
            items = self.iter_from_dir(input_data) if is_dir else self.iter_from_json_file(input_data)
            convert_to_asr_json_manifest(
                items, output_data, data_key=self._data_keys[0], project_dir=self.project_dir,
                upload_dir=self.upload_dir)

    def _get_data_keys_and_output_tags(self, output_tags=None):
        data_keys = set()
        output_tag_names = []
        if output_tags is not None:
            for tag in output_tags:
                if tag not in self._schema:
                    logger.warning(
                        'Specified tag "{tag}" not found in config schema: '
                        'available options are {schema_keys}'.format(
                            tag=tag, schema_keys=str(list(self._schema.keys()))))
        for name, info in self._schema.items():
            if output_tags is not None and name not in output_tags:
                continue
            data_keys |= set(map(itemgetter('value'), info['inputs']))
            output_tag_names.append(name)

        return list(data_keys), output_tag_names

    def _get_supported_formats(self):
        if len(self._data_keys) > 1:
            return [Format.JSON.name, Format.JSON_MIN.name, Format.CSV.name, Format.TSV.name]
        output_tag_types = set()
        input_tag_types = set()
        for info in self._schema.values():
            output_tag_types.add(info['type'])
            for input_tag in info['inputs']:
                input_tag_types.add(input_tag['type'])

        all_formats = [f.name for f in Format]
        if not ('Text' in input_tag_types and 'Labels' in output_tag_types):
            all_formats.remove(Format.CONLL2003.name)
        if not ('Image' in input_tag_types and 'RectangleLabels' in output_tag_types):
            all_formats.remove(Format.VOC.name)
        if not ('Image' in input_tag_types and ('RectangleLabels' in output_tag_types or
                                                'PolygonLabels' in output_tag_types)):
            all_formats.remove(Format.COCO.name)
        if not ('Image' in input_tag_types and ('BrushLabels' in output_tag_types or 'brushlabels' in output_tag_types)):
            all_formats.remove(Format.BRUSH_TO_NUMPY.name)
            all_formats.remove(Format.BRUSH_TO_PNG.name)
        if not (('Audio' in input_tag_types or 'AudioPlus' in input_tag_types) and 'TextArea' in output_tag_types):
            all_formats.remove(Format.ASR_MANIFEST.name)

        return all_formats

    @property
    def supported_formats(self):
        return self._supported_formats

    def iter_from_dir(self, input_dir):
        if not os.path.exists(input_dir):
            raise FileNotFoundError('{input_dir} doesn\'t exist'.format(input_dir=input_dir))
        for json_file in glob(os.path.join(input_dir, '*.json')):
            for item in self.iter_from_json_file(json_file):
                if item:
                    yield item

    def iter_from_json_file(self, json_file):
        with io.open(json_file, encoding='utf8') as f:
            data = json.load(f)
            if isinstance(data, Mapping):
                yield self.load_from_dict(data)
            elif isinstance(data, list):
                for item in data:
                    prepared_item = self.load_from_dict(item)
                    if prepared_item is not None:
                        yield prepared_item

    def load_from_dict(self, d):
        has_annotations = 'completions' in d or 'annotations' in d
        if not has_annotations and 'result' not in d:
            raise KeyError('Each annotation dict item should contain "annotations" or "result" key, '
                           'where value is list of dicts')
        result = []
        if has_annotations:
            # get last not skipped completion and make result from it
            annotations = d.get('annotations')
            if annotations is None:
                annotations = d['completions']
            tmp = list(filter(lambda x: not (x.get('skipped', False) or x.get('was_cancelled', False)), annotations))
            if len(tmp) > 0:
                # TODO: only one annotation per task is supported for non full JSON formats
                result = sorted(tmp, key=lambda x: x.get('created_at', 0), reverse=True)[0]['result']
            else:
                return None

        elif 'result' in d:
            result = d['result']
        inputs = d['data']
        outputs = defaultdict(list)
        for r in result:
            if 'from_name' in r and r['from_name'] in self._output_tags:
                v = deepcopy(r['value'])
                v['type'] = self._schema[r['from_name']]['type']
                if 'original_width' in r:
                    v['original_width'] = r['original_width']
                if 'original_height' in r:
                    v['original_height'] = r['original_height']
                outputs[r['from_name']].append(v)
        return {
            'id': d['id'],
            'input': inputs,
            'output': outputs
        }

    def _check_format(self, fmt):
        pass

    def _prettify(self, v):
        out = []
        tag_type = None
        for i in v:
            j = deepcopy(i)
            tag_type = j.pop('type')
            if tag_type == 'Choices' and len(j['choices']) == 1:
                out.append(j['choices'][0])
            else:
                out.append(j)
        return out[0] if tag_type == 'Choices' and len(out) == 1 else out

    def convert_to_json(self, input_data, output_dir, is_dir=True):
        self._check_format(Format.JSON)
        ensure_dir(output_dir)
        output_file = os.path.join(output_dir, 'result.json')
        records = []
        if is_dir:
            for json_file in glob(os.path.join(input_data, '*.json')):
                with io.open(json_file, encoding='utf8') as f:
                    records.append(json.load(f))
            with io.open(output_file, mode='w', encoding='utf8') as fout:
                json.dump(records, fout, indent=2, ensure_ascii=False)
        else:
            copy2(input_data, output_file)

    def convert_to_json_min(self, input_data, output_dir, is_dir=True):
        self._check_format(Format.JSON_MIN)
        ensure_dir(output_dir)
        output_file = os.path.join(output_dir, 'result.json')
        records = []
        item_iterator = self.iter_from_dir if is_dir else self.iter_from_json_file
        for item in item_iterator(input_data):
            record = deepcopy(item['input'])
            if item.get('id') is not None:
                record['id'] = item['id']
            for name, value in item['output'].items():
                record[name] = self._prettify(value)
            records.append(record)

        with io.open(output_file, mode='w', encoding='utf8') as fout:
            json.dump(records, fout, indent=2, ensure_ascii=False)

    def convert_to_csv(self, input_data, output_dir, is_dir=True, **kwargs):
        self._check_format(Format.CSV)
        ensure_dir(output_dir)
        output_file = os.path.join(output_dir, 'result.csv')
        records = []
        item_iterator = self.iter_from_dir if is_dir else self.iter_from_json_file
        for item in item_iterator(input_data):
            record = deepcopy(item['input'])
            if item.get('id') is not None:
                record['id'] = item['id']
            for name, value in item['output'].items():
                pretty_value = self._prettify(value)
                record[name] = pretty_value if isinstance(pretty_value, str) else json.dumps(pretty_value)
            records.append(record)

        pd.DataFrame.from_records(records).to_csv(output_file, index=False, **kwargs)

    def convert_to_conll2003(self, input_data, output_dir, is_dir=True):
        self._check_format(Format.CONLL2003)
        ensure_dir(output_dir)
        output_file = os.path.join(output_dir, 'result.conll')
        data_key = self._data_keys[0]
        with io.open(output_file, 'w', encoding='utf8') as fout:
            fout.write('-DOCSTART- -X- O\n')
            item_iterator = self.iter_from_dir if is_dir else self.iter_from_json_file

            for item in item_iterator(input_data):
                tokens, tags = create_tokens_and_tags(
                    text=item['input'][data_key],
                    spans=next(iter(item['output'].values()), None)
                )
                for token, tag in zip(tokens, tags):
                    fout.write('{token} -X- _ {tag}\n'.format(token=token, tag=tag))
                fout.write('\n')

    def convert_to_coco(self, input_data, output_dir, output_image_dir=None, is_dir=True):
        self._check_format(Format.COCO)
        ensure_dir(output_dir)
        output_file = os.path.join(output_dir, 'result.json')
        if output_image_dir is not None:
            ensure_dir(output_image_dir)
        else:
            output_image_dir = os.path.join(output_dir, 'images')
            os.makedirs(output_image_dir, exist_ok=True)
        images, annotations = [], []
        categories = [
            {"id": n, "name": x} for n, x in enumerate(self._schema["label"]["labels"])
        ]
        category_name_to_id = {x["name"]: x["id"] for x in categories}

        data_key = self._data_keys[0]
        item_iterator = self.iter_from_dir(input_data) if is_dir else self.iter_from_json_file(input_data)
        for item_idx, item in enumerate(item_iterator):
            if not item['output']:
                logger.warning('No annotations found for item #' + str(item_idx))
                continue
            image_path = item['input'][data_key]
            if not os.path.exists(image_path):
                try:
                    image_path = download(image_path, output_image_dir, project_dir=self.project_dir, return_relative_path=True)
                except:
                    logger.error('Unable to download {image_path}. The item {item} will be skipped'.format(
                        image_path=image_path, item=item
                    ), exc_info=True)
            labels = next(iter(item['output'].values()))
            if len(labels) == 0:
                logger.error('Empty bboxes.')
                continue
            width, height = labels[0]['original_width'], labels[0]['original_height']
            image_id = len(images)
            images.append({
                'width': width,
                'height': height,
                'id': image_id,
                'file_name': image_path
            })

            for label in labels:
                if 'rectanglelabels' in label:
                    category_name = label['rectanglelabels'][0]
                elif 'polygonlabels' in label:
                    category_name = label['polygonlabels'][0]
                else:
                    raise ValueError("Unknown label type")

                category_id = category_name_to_id[category_name]

                annotation_id = len(annotations)

                if "rectanglelabels" in label:
                    x = int(label['x'] / 100 * width)
                    y = int(label['y'] / 100 * height)
                    w = int(label['width'] / 100 * width)
                    h = int(label['height'] / 100 * height)

                    annotations.append({
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': category_id,
                        'segmentation': [],
                        'bbox': [x, y, w, h],
                        'ignore': 0,
                        'iscrowd': 0,
                        'area': w * h
                    })
                elif "polygonlabels" in label:
                    points_abs = [(x / 100 * width, y / 100 * height) for x, y in label["points"]]
                    x, y = zip(*points_abs)

                    annotations.append({
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': category_id,
                        'segmentation': [[coord for point in points_abs for coord in point]],
                        'bbox': get_polygon_bounding_box(x, y),
                        'ignore': 0,
                        'iscrowd': 0,
                        'area': get_polygon_area(x, y)
                    })
                else:
                    raise ValueError("Unknown label type")

        with io.open(output_file, mode='w', encoding='utf8') as fout:
            json.dump({
                'images': images,
                'categories': categories,
                'annotations': annotations,
                'info': {
                    'year': datetime.now().year,
                    'version': '1.0',
                    'contributor': 'Label Studio'
                }
            }, fout, indent=2)

    def convert_to_voc(self, input_data, output_dir, output_image_dir=None, is_dir=True):

        ensure_dir(output_dir)
        if output_image_dir is not None:
            ensure_dir(output_image_dir)
            output_image_dir_rel = output_image_dir
        else:
            output_image_dir = os.path.join(output_dir, 'images')
            os.makedirs(output_image_dir, exist_ok=True)
            output_image_dir_rel = 'images'

        def create_child_node(doc, tag, attr, parent_node):
            child_node = doc.createElement(tag)
            text_node = doc.createTextNode(attr)
            child_node.appendChild(text_node)
            parent_node.appendChild(child_node)

        data_key = self._data_keys[0]
        item_iterator = self.iter_from_dir(input_data) if is_dir else self.iter_from_json_file(input_data)
        for item_idx, item in enumerate(item_iterator):
            if not item['output']:
                logger.warning('No annotations found for item #' + str(item_idx))
                continue
            image_path = item['input'][data_key]
            annotations_dir = os.path.join(output_dir, 'Annotations')
            if not os.path.exists(annotations_dir):
                os.makedirs(annotations_dir)
            if not os.path.exists(image_path):
                try:
                    image_path = download(
                        image_path, output_image_dir, project_dir=self.project_dir,
                        upload_dir=self.upload_dir, return_relative_path=True)
                except:
                    logger.error('Unable to download {image_path}. The item {item} will be skipped'.format(
                        image_path=image_path, item=item), exc_info=True)
                    # On error, use default number of channels
                    channels = 3
                else:
                    full_image_path = os.path.join(output_image_dir, os.path.basename(image_path))
                    # retrieve number of channels from downloaded image
                    _, _, channels = get_image_size_and_channels(full_image_path)

            bboxes = next(iter(item['output'].values()))
            if len(bboxes) == 0:
                logger.error('Empty bboxes.')
                continue

            width, height = bboxes[0]['original_width'], bboxes[0]['original_height']

            image_name = os.path.splitext(os.path.basename(image_path))[0]
            xml_filepath = os.path.join(annotations_dir, image_name + '.xml')

            my_dom = xml.dom.getDOMImplementation()
            doc = my_dom.createDocument(None, 'annotation', None)
            root_node = doc.documentElement
            create_child_node(doc, 'folder', output_image_dir_rel, root_node)
            create_child_node(doc, 'filename', image_name, root_node)

            source_node = doc.createElement('source')
            create_child_node(doc, 'database', 'MyDatabase', source_node)
            create_child_node(doc, 'annotation', 'COCO2017', source_node)
            create_child_node(doc, 'image', 'flickr', source_node)
            create_child_node(doc, 'flickrid', 'NULL', source_node)
            root_node.appendChild(source_node)
            owner_node = doc.createElement('owner')
            create_child_node(doc, 'flickrid', 'NULL', owner_node)
            create_child_node(doc, 'name', 'Label Studio', owner_node)
            root_node.appendChild(owner_node)
            size_node = doc.createElement('size')
            create_child_node(doc, 'width', str(width), size_node)
            create_child_node(doc, 'height', str(height), size_node)
            create_child_node(doc, 'depth', str(channels), size_node)
            root_node.appendChild(size_node)
            create_child_node(doc, 'segmented', '0', root_node)


            for bbox in bboxes:
                name = bbox['rectanglelabels'][0]
                x = int(bbox['x'] / 100 * width)
                y = int(bbox['y'] / 100 * height)
                w = int(bbox['width'] / 100 * width)
                h = int(bbox['height'] / 100 * height)

                object_node = doc.createElement('object')
                create_child_node(doc, 'name', name, object_node)
                create_child_node(doc, 'pose', 'Unspecified', object_node)
                create_child_node(doc, 'truncated', '0', object_node)
                create_child_node(doc, 'difficult', '0', object_node)
                bndbox_node = doc.createElement('bndbox')
                create_child_node(doc, 'xmin', str(x), bndbox_node)
                create_child_node(doc, 'ymin', str(y), bndbox_node)
                create_child_node(doc, 'xmax', str(x + w), bndbox_node)
                create_child_node(doc, 'ymax', str(y + h), bndbox_node)
                object_node.appendChild(bndbox_node)
                root_node.appendChild(object_node)

            with io.open(xml_filepath, mode='w', encoding='utf8') as fout:
                doc.writexml(fout, addindent='' * 4, newl='\n', encoding='utf-8')
