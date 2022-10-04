import os
import json  # better to use "imports ujson as json" for the best performance
import uuid
import logging
from PIL import Image

from label_studio_converter.utils import ExpandFullPath
from label_studio_converter.imports.label_config import generate_label_config

logger = logging.getLogger('root')


def new_task(out_type, root_url, file_name):
    return {
        "data": {
            "image": os.path.join(root_url, file_name)
        },
        # 'annotations' or 'predictions'
        out_type: [
            {
                "result": [],
                "ground_truth": False,
            }
        ]
    }


def create_bbox(annotation, categories, from_name, image_height, image_width, to_name):
    label = categories[int(annotation['category_id'])]
    x, y, width, height = annotation['bbox']
    x, y, width, height = float(x), float(y), float(width), float(height)
    item = {
        "id": uuid.uuid4().hex[0:10],
        "type": "rectanglelabels",
        "value": {
            "x": x / image_width * 100.0,
            "y": y / image_height * 100.0,
            "width": width / image_width * 100.0,
            "height": height / image_height * 100.0,
            "rotation": 0,
            "rectanglelabels": [
                label
            ]
        },
        "to_name": to_name,
        "from_name": from_name,
        "image_rotation": 0,
        "original_width": image_width,
        "original_height": image_height
    }
    return item


def create_keypoints(annotation, categories, from_name, to_name, image_height, image_width, point_width):
    label = categories[int(annotation['category_id'])]
    points = annotation['keypoints']
    items = []

    for i in range(0, len(points), 3):
        x, y, v = points[i:i+3]  # x, y, visibility
        x, y, v = float(x), float(y), int(v)
        item = {
            "id": uuid.uuid4().hex[0:10],
            "type": "keypointlabels",
            "value": {
                "x": x / image_width * 100.0,
                "y": y / image_height * 100.0,
                "width": point_width,
                "keypointlabels": [
                    label
                ]
            },
            "to_name": to_name,
            "from_name": from_name,
            "image_rotation": 0,
            "original_width": image_width,
            "original_height": image_height
        }

        # visibility
        if v < 2:
            item['value']['hidden'] = True

        items.append(item)
    return items


def convert_coco_to_ls(input_file, out_file,
                       to_name='image', from_name='label', out_type="annotations",
                       image_root_url='/data/local-files/?d=',
                       use_super_categories=False,
                       point_width=1.0):

    """ Convert COCO labeling to Label Studio JSON

    :param input_file: file with COCO json
    :param out_file: output file with Label Studio JSON tasks
    :param to_name: object name from Label Studio labeling config
    :param from_name: control tag name from Label Studio labeling config
    :param out_type: annotation type - "annotations" or "predictions"
    :param image_root_url: root URL path where images will be hosted, e.g.: http://example.com/images
    :param use_super_categories: use super categories from categories if they are presented
    :param point_width: key point width
    """

    tasks = {}  # image_id => task
    logger.info('Reading COCO notes and categories from %s', input_file)

    with open(input_file, encoding='utf8') as f:
        coco = json.load(f)

    # build categories => labels dict
    new_categories = {}
    # list to dict conversion: [...] => {category_id: category_item}
    categories = {int(category['id']): category for category in coco['categories']}
    ids = sorted(categories.keys())  # sort labels by their origin ids

    for i in ids:
        name = categories[i]['name']
        if use_super_categories and 'supercategory' in categories[i]:
            name = categories[i]['supercategory'] + ':' + name
        new_categories[i] = name

    # mapping: id => category name
    categories = new_categories

    # mapping: image id => image
    images = {item['id']: item for item in coco['images']}

    logger.info(f'Found {len(categories)} categories, {len(images)} images and {len(coco["annotations"])} annotations')

    # flags for labeling config composing
    segmentation = bbox = keypoints = rle = False
    segmentation_once = bbox_once = keypoints_once = rle_once = False
    rectangles_from_name, keypoints_from_name = from_name + '_rectangles', from_name + '_keypoints'
    tags = {}

    for i, annotation in enumerate(coco['annotations']):
        segmentation |= 'segmentation' in annotation
        bbox |= 'bbox' in annotation
        keypoints |= 'keypoints' in annotation
        rle |= annotation.get('iscrowd') == 1  # 0 - polygons are in segmentation, otherwise rle

        if rle and not rle_once:  # not supported
            logger.error('RLE in segmentation is not yet supported in COCO')
            rle_once = True
        if keypoints and not keypoints_once:
            logger.warning('Keypoints are partially supported without skeletons')
            tags.update({keypoints_from_name: 'KeyPointLabels'})
            keypoints_once = True
        if segmentation and not segmentation_once:  # not supported
            logger.error('Segmentation is not yet supported in COCO')
            segmentation_once = True
        if bbox and not bbox_once:
            tags.update({rectangles_from_name: 'RectangleLabels'})
            bbox_once = True

        # read image sizes
        image_id = annotation['image_id']
        image = images[image_id]
        image_file_name, image_width, image_height = image['file_name'], image['width'], image['height']

        # get or create new task
        task = tasks[image_id] if image_id in tasks else new_task(out_type, image_root_url, image_file_name)

        if 'bbox' in annotation:
            item = create_bbox(annotation, categories, rectangles_from_name, image_height, image_width, to_name)
            task[out_type][0]['result'].append(item)

        if 'keypoints' in annotation:
            items = create_keypoints(annotation, categories, keypoints_from_name, to_name,
                                     image_height, image_width, point_width)
            task[out_type][0]['result'] += items

        tasks[image_id] = task

    # generate and save labeling config
    label_config_file = out_file.replace('.json', '') + '.label_config.xml'
    generate_label_config(categories, tags, to_name, from_name, label_config_file)

    if len(tasks) > 0:
        tasks = [tasks[key] for key in sorted(tasks.keys())]
        logger.info('Saving Label Studio JSON to %s', out_file)
        with open(out_file, 'w') as out:
            json.dump(tasks, out)

        print('\n'
              f'  1. Create a new project in Label Studio\n'
              f'  2. Use Labeling Config from "{label_config_file}"\n'
              f'  3. Setup serving for images [e.g. you can use Local Storage (or others):\n'
              f'     https://labelstud.io/guide/storage.html#Local-storage]\n'
              f'  4. Import "{out_file}" to the project\n')
    else:
        logger.error('No labels converted')


def add_parser(subparsers):
    coco = subparsers.add_parser('coco')

    coco.add_argument(
        '-i', '--input', dest='input', required=True,
        help='directory with COCO where images, labels, notes.json are located',
        action=ExpandFullPath
    )
    coco.add_argument(
        '-o', '--output', dest='output',
        help='output file with Label Studio JSON tasks',
        default='output.json',
        action=ExpandFullPath
    )
    coco.add_argument(
        '--to-name', dest='to_name',
        help='object name from Label Studio labeling config',
        default='image',
    )
    coco.add_argument(
        '--from-name', dest='from_name',
        help='control tag name from Label Studio labeling config',
        default='label',
    )
    coco.add_argument(
        '--out-type', dest='out_type',
        help='annotation type - "annotations" or "predictions"',
        default='annotations',
    )
    coco.add_argument(
        '--image-root-url', dest='image_root_url',
        help='root URL path where images will be hosted, e.g.: http://example.com/images',
        default='/data/local-files/?d=',
    )
    coco.add_argument(
        '--point-width', dest='point_width',
        help='key point width (size)',
        default=1.0,
        type=float
    )

