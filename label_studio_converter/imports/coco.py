import os
import json  # better to use "imports ujson as json" for the best performance
import uuid
import logging
from PIL import Image

from label_studio_converter.utils import ExpandFullPath
from label_studio_converter.imports.label_config import generate_label_config

logger = logging.getLogger('root')


def convert_coco_to_ls(input_file, out_file,
                       to_name='image', from_name='label', out_type="annotations",
                       image_root_url='/data/local-files/?d=', image_ext='.jpg',
                       use_super_categories=False):

    """ Convert COCO labeling to Label Studio JSON

    :param input_file: file with COCO json
    :param out_file: output file with Label Studio JSON tasks
    :param to_name: object name from Label Studio labeling config
    :param from_name: control tag name from Label Studio labeling config
    :param out_type: annotation type - "annotations" or "predictions"
    :param image_root_url: root URL path where images will be hosted, e.g.: http://example.com/images
    :param image_ext: image extension to search: .jpg, .png
    :param use_super_categories: use super categories from categories if they are presented
    """

    tasks = []
    logger.info('Reading COCO notes and categories from %s', input_file)

    with open(input_file, encoding='utf8') as f:
        coco = json.load(f)

    # build categories=>labels dict
    categories = coco['categories']
    new_categories = {}
    ids = sorted(category['id'] for category in categories)
    for i in ids:
        name = categories[i]['name']
        if use_super_categories and 'supercategory' in categories[i]:
            name = categories[i]['supercategory'] + ':' + name

        new_categories[i] = name
    categories = new_categories

    logger.info(f'Found {len(categories)} categories')

    # generate and save labeling config
    label_config_file = out_file.replace('.json', '') + '.label_config.xml'
    generate_label_config(categories, to_name, from_name, label_config_file)

    # labels, one label per image
    logger.info('Converting labels from %s', labels_dir)

    for annotation in coco['annotations']:
        image_file = annotation['']

        task = {
            # 'annotations' or 'predictions'
            out_type: [
                {
                    "result": [],
                    "ground_truth": False,
                }
            ],
            "data": {
                "image": os.path.join(image_root_url, image_file_base)
            }
        }

        # read image sizes
        im = Image.open(image_file)
        image_width, image_height = im.size

        # convert all bounding boxes to Label Studio Results
        lines = coco['annotations']
        for line in lines:
            label_id, x, y, width, height = line.split()
            x, y, width, height = float(x), float(y), float(width), float(height)
            item = {
                "id": uuid.uuid4().hex[0:10],
                "type": "rectanglelabels",
                "value": {
                    "x": (x-width/2) * 100,
                    "y": (y-height/2) * 100,
                    "width": width * 100,
                    "height": height * 100,
                    "rotation": 0,
                    "rectanglelabels": [
                        categories[int(label_id)]
                    ]
                },
                "to_name": to_name,
                "from_name": from_name,
                "image_rotation": 0,
                "original_width": image_width,
                "original_height": image_height
            }
            task['annotations'][0]['result'].append(item)

        tasks.append(task)

    if len(tasks) > 0:
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
        '--image-ext', dest='image_ext',
        help='image extension to search: .jpg, .png',
        default='.jpg',
    )
