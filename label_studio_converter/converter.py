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
    get_polygon_area, get_polygon_bounding_box, _get_annotator
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
    YOLO = 11

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
            'description': "List of items in raw JSON format stored in one JSON file. Use to export both the data "
                           "and the annotations for a dataset. It's Label Studio Common Format",
            'link': 'https://labelstud.io/guide/export.html#JSON'
        },
        Format.JSON_MIN: {
            'title': 'JSON-MIN',
            'description': 'List of items where only "from_name", "to_name" values from the raw JSON format are '
                           'exported. Use to export only the annotations for a dataset.',
            'link': 'https://labelstud.io/guide/export.html#JSON-MIN',
        },
        Format.CSV: {
            'title': 'CSV',
            'description': 'Results are stored as comma-separated values with the column names specified by the '
                           'values of the "from_name" and "to_name" fields.',
            'link': 'https://labelstud.io/guide/export.html#CSV'
        },
        Format.TSV: {
            'title': 'TSV',
            'description': 'Results are stored in tab-separated tabular file with column names specified by '
                           '"from_name" "to_name" values',
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
            'description': 'Popular machine learning format used by the COCO dataset for object detection and image '
                           'segmentation tasks with polygons and rectangles.',
            'link': 'https://labelstud.io/guide/export.html#COCO',
            'tags': ['image segmentation', 'object detection']
        },
        Format.VOC: {
            'title': 'Pascal VOC XML',
            'description': 'Popular XML format used for object detection and polygon image segmentation tasks.',
            'link': 'https://labelstud.io/guide/export.html#Pascal-VOC-XML',
            'tags': ['image segmentation', 'object detection']
        },
        Format.YOLO: {
            'title': 'YOLO',
            'description': 'Popular TXT format is created for each image file. Each txt file contains annotations for '
                           'the corresponding image file, that is object class, object coordinates, height & width.',
            'link': 'https://labelstud.io/guide/export.html#YOLO',
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
            'description': 'Export audio transcription labels for automatic speech recognition as the JSON manifest '
                           'format expected by NVIDIA NeMo models.',
            'link': 'https://labelstud.io/guide/export.html#ASR-MANIFEST',
            'tags': ['speech recognition']
        }
    }

    def all_formats(self):
        return self._FORMAT_INFO

    def __init__(self, config, project_dir, output_tags=None, upload_dir=None, download_resources=True):
        self.project_dir = project_dir
        self.upload_dir = upload_dir
        self.download_resources = download_resources
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
        elif format == Format.YOLO:
            image_dir = kwargs.get('image_dir')
            label_dir = kwargs.get('label_dir')
            self.convert_to_yolo(input_data, output_data, output_image_dir=image_dir,
                                 output_label_dir=label_dir, is_dir=is_dir)
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
                upload_dir=self.upload_dir, download_resources=self.download_resources)

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
        if not ('Image' in input_tag_types and ('RectangleLabels' in output_tag_types or
                                                'Rectangle' in output_tag_types and 'Labels' in output_tag_types)):
            all_formats.remove(Format.VOC.name)
            all_formats.remove(Format.YOLO.name)
        if not ('Image' in input_tag_types and ('RectangleLabels' in output_tag_types or
                                                'PolygonLabels' in output_tag_types) or
                                                'Rectangle' in output_tag_types and 'Labels' in output_tag_types or
                                                'PolygonLabels' in output_tag_types and 'Labels' in output_tag_types):

            all_formats.remove(Format.COCO.name)
        if not ('Image' in input_tag_types and ('BrushLabels' in output_tag_types or 'brushlabels' in output_tag_types or
                                                'Brush' in output_tag_types and 'Labels' in output_tag_types)):
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
        """ Extract annotation results from json file

        param json_file: path to task list or dict with annotations
        """
        with io.open(json_file, encoding='utf8') as f:
            data = json.load(f)
            # one task
            if isinstance(data, Mapping):
                for item in self.annotation_result_from_task(data):
                    yield item

            # many tasks
            elif isinstance(data, list):
                for task in data:
                    for item in self.annotation_result_from_task(task):
                        if item is not None:
                            yield item

    def annotation_result_from_task(self, task):
        has_annotations = 'completions' in task or 'annotations' in task
        if not has_annotations:
            raise KeyError('Each task dict item should contain "annotations" or "completions" [deprecated],'
                           'where value is list of dicts')

        # get last not skipped completion and make result from it
        annotations = task['annotations'] if 'annotations' in task else task['completions']
        
        # skip cancelled annotations
        cancelled = lambda x: not (x.get('skipped', False) or x.get('was_cancelled', False))
        annotations = list(filter(cancelled, annotations))
        if not annotations:
            return None
        
        # sort by creation time
        annotations = sorted(annotations, key=lambda x: x.get('created_at', 0), reverse=True)

        for annotation in annotations:
            inputs = task['data']
            result = annotation['result']
            outputs = defaultdict(list)

            # get results only as output
            for r in result:
                if 'from_name' in r and r['from_name'] in self._output_tags:
                    v = deepcopy(r['value'])
                    v['type'] = self._schema[r['from_name']]['type']
                    if 'original_width' in r:
                        v['original_width'] = r['original_width']
                    if 'original_height' in r:
                        v['original_height'] = r['original_height']
                    outputs[r['from_name']].append(v)

            yield {
                'id': task['id'],
                'input': inputs,
                'output': outputs,
                'completed_by': annotation.get('completed_by', {}),
                'annotation_id': annotation.get('id'),
                'created_at': annotation.get('created_at'),
                'updated_at': annotation.get('updated_at'),
                'lead_time': annotation.get('lead_time')
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
            elif tag_type == 'TextArea' and len(j['text']) == 1:
                out.append(j['text'][0])
            else:
                out.append(j)
        return out[0] if tag_type in ('Choices', 'TextArea') and len(out) == 1 else out

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
            record['annotator'] = _get_annotator(item)
            record['annotation_id'] = item['annotation_id']
            record['created_at'] = item['created_at']
            record['updated_at'] = item['updated_at']
            record['lead_time'] = item['lead_time']
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
            record['annotator'] = _get_annotator(item)
            record['annotation_id'] = item['annotation_id']
            record['created_at'] = item['created_at']
            record['updated_at'] = item['updated_at']
            record['lead_time'] = item['lead_time']
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
        images, categories, annotations = [], [], []
        categories, category_name_to_id = self._get_labels()
        data_key = self._data_keys[0]
        item_iterator = self.iter_from_dir(input_data) if is_dir else self.iter_from_json_file(input_data)
        for item_idx, item in enumerate(item_iterator):
            if not item['output']:
                logger.warning('No annotations found for item #' + str(item_idx))
                continue
            image_path = item['input'][data_key]
            if not os.path.exists(image_path):
                try:
                    image_path = download(image_path, output_image_dir, project_dir=self.project_dir,
                                          return_relative_path=True, upload_dir=self.upload_dir,
                                          download_resources=self.download_resources)
                except:
                    logger.error('Unable to download {image_path}. The item {item} will be skipped'.format(
                        image_path=image_path, item=item
                    ), exc_info=True)

            # concatentate results over all tag names
            labels = []
            for key in item['output']:
                labels += item['output'][key]

            if len(labels) == 0:
                logger.warning(f'Empty bboxes for {item["output"]}')
                continue

            first = True

            for label in labels:
                if 'rectanglelabels' in label:
                    category_name = label['rectanglelabels'][0]
                elif 'polygonlabels' in label:
                    category_name = label['polygonlabels'][0]
                elif 'labels' in label:
                    category_name = label['labels'][0]
                else:
                    logger.warning("Unknown label type: " + str(label))
                    continue

                # get image sizes
                if first:
                    width, height = label['original_width'], label['original_height']
                    image_id = len(images)
                    images.append({
                        'width': width,
                        'height': height,
                        'id': image_id,
                        'file_name': image_path
                    })
                    first = False

                '''if category_name not in category_name_to_id:
                    category_id = len(categories)
                    category_name_to_id[category_name] = category_id
                    categories.append({
                        'id': category_id,
                        'name': category_name,
                        'supercategory': category_name
                    })'''
                category_id = category_name_to_id[category_name]

                annotation_id = len(annotations)

                if 'rectanglelabels' in label or 'labels' in label:
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
                        'area': w * h,
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

                if os.getenv('LABEL_STUDIO_FORCE_ANNOTATOR_EXPORT'):
                    annotations[-1].update({'annotator': _get_annotator(item)})

        with io.open(output_file, mode='w', encoding='utf8') as fout:
            json.dump({
                'images': images,
                'categories': categories,
                'annotations': annotations,
                'info': {
                    'year': datetime.now().year,
                    'version': '1.0',
                    'description': '',
                    'contributor': 'Label Studio',
                    'url': '',
                    'date_created': str(datetime.now())
                }
            }, fout, indent=2)

    def convert_to_yolo(self, input_data, output_dir, output_image_dir=None, output_label_dir=None, is_dir=True):
        self._check_format(Format.YOLO)
        ensure_dir(output_dir)
        notes_file = os.path.join(output_dir, 'notes.json')
        class_file = os.path.join(output_dir, 'classes.txt')
        if output_image_dir is not None:
            ensure_dir(output_image_dir)
        else:
            output_image_dir = os.path.join(output_dir, 'images')
            os.makedirs(output_image_dir, exist_ok=True)
        if output_label_dir is not None:
            ensure_dir(output_label_dir)
        else:
            output_label_dir = os.path.join(output_dir, 'labels')
            os.makedirs(output_label_dir, exist_ok=True)
        categories = []
        category_name_to_id = {}
        data_key = self._data_keys[0]
        item_iterator = self.iter_from_dir(input_data) if is_dir else self.iter_from_json_file(input_data)
        for item_idx, item in enumerate(item_iterator):
            if not item['output']:
                logger.warning('No completions found for item #' + str(item_idx))
                continue
            image_path = item['input'][data_key]
            if not os.path.exists(image_path):
                try:
                    image_path = download(image_path, output_image_dir, project_dir=self.project_dir,
                                          return_relative_path=True, upload_dir=self.upload_dir,
                                          download_resources=self.download_resources)
                except:
                    logger.error('Unable to download {image_path}. The item {item} will be skipped'.format(
                        image_path=image_path, item=item
                    ), exc_info=True)

            # concatentate results over all tag names
            labels = []
            for key in item['output']:
                labels += item['output'][key]

            if len(labels) == 0:
                logger.warning(f'Empty bboxes for {item["output"]}')
                continue

            label_path = os.path.join(output_label_dir, os.path.splitext(os.path.basename(image_path))[0]+'.txt')
            annotations = []
            for label in labels:
                if 'rectanglelabels' in label:
                    category_name = label['rectanglelabels'][0]
                elif 'labels' in label:
                    category_name = label['labels'][0]
                else:
                    logger.warning(f"Unknown label type {label}")
                    continue
                if category_name not in category_name_to_id:
                    category_id = len(categories)
                    category_name_to_id[category_name] = category_id
                    categories.append({
                        'id': category_id,
                        'name': category_name
                    })
                category_id = category_name_to_id[category_name]

                if "rectanglelabels" in label or 'labels' in label:
                    x = (label['x'] + label['width']/2) / 100
                    y = (label['y'] + label['height']/2) / 100
                    w = label['width'] / 100
                    h = label['height'] / 100
                    annotations.append([category_id, x, y, w, h])
                else:
                    raise ValueError(f"Unknown label type {label}")
            with open(label_path, 'w') as f:
                for annotation in annotations:
                    for idx, l in enumerate(annotation):
                        if idx == len(annotation) -1:
                            f.write(f"{l}\n")
                        else:
                            f.write(f"{l} ")
        with open(class_file, 'w', encoding='utf8') as f:
            for c in categories:
                f.write(c['name']+'\n')
        with io.open(notes_file, mode='w', encoding='utf8') as fout:
            json.dump({
                'categories': categories,
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

            channels = 3
            if not os.path.exists(image_path):
                try:
                    image_path = download(
                        image_path, output_image_dir, project_dir=self.project_dir,
                        upload_dir=self.upload_dir, return_relative_path=True, download_resources=self.download_resources)
                except:
                    logger.error('Unable to download {image_path}. The item {item} will be skipped'.format(
                        image_path=image_path, item=item), exc_info=True)
                else:
                    full_image_path = os.path.join(output_image_dir, os.path.basename(image_path))
                    # retrieve number of channels from downloaded image
                    _, _, channels = get_image_size_and_channels(full_image_path)

            bboxes = next(iter(item['output'].values()))

            # concatentate results over all tag names
            bboxes = []
            for key in item['output']:
                bboxes += item['output'][key]

            if len(bboxes) == 0:
                logger.warning(f'Empty bboxes for {item["output"]}')
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
            create_child_node(doc, 'annotator', _get_annotator(item, ''), source_node)
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
                key = 'rectanglelabels' if 'rectanglelabels' in bbox else ('labels' if 'labels' in bbox else None)
                if key is None:
                    continue

                name = bbox[key][0]
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

    def _get_labels(self):
        labels = set()
        categories = list()
        category_name_to_id = dict()

        for name, info in self._schema.items():
            labels |= set(info['labels'])
            attrs = info['labels_attrs']
            for label in attrs:
                if attrs[label].get('category'):
                    categories.append({
                        'id': attrs[label].get('category'),
                        'name': label
                    })
                    category_name_to_id[label] = attrs[label].get('category')
        labels_to_add = set(labels) - set(list(category_name_to_id.keys()))
        labels_to_add = sorted(list(labels_to_add))
        idx = 0
        while idx in list(category_name_to_id.values()):
            idx += 1
        for label in labels_to_add:
            categories.append({
                'id': idx,
                'name': label
            })
            category_name_to_id[label] = idx
            idx += 1
            while idx in list(category_name_to_id.values()):
                idx += 1
        return categories, category_name_to_id
