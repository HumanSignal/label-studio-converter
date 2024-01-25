import os
import shutil
from pathlib import Path
import json  # better to use "imports ujson as json" for the best performance

import uuid
import logging

from PIL import Image
from typing import Optional, Tuple
from urllib.request import (
    pathname2url,
)  # for converting "+","*", etc. in file paths to appropriate urls

from label_studio_converter.utils import ExpandFullPath
from label_studio_converter.imports.label_config import generate_label_config

logger = logging.getLogger('root')

def get_data(input_dir, img_exts):
    get_labels = lambda files: list( filter(lambda fn: fn.endswith('.txt') and 'classes.txt' not in fn, files if type(files)==list else os.listdir(files)) )  
    get_images = lambda files: list( filter(lambda fn: any([fn.endswith(img_ext) for img_ext in img_exts]), files if type(files)==list else os.listdir(files)) )  
    images, labels = [], []
    image_labels = {}
    for dir_pth, dir_names, files in os.walk(input_dir):
        if Path(dir_pth) == Path(input_dir):
            continue         # skip input_dir. data should be at least one level in from input_dir
        dir_imgs, dir_lbls = get_images( files ), get_labels( files )
        if len(dir_imgs) > 0:
            [images.append(f'{dir_pth}/{img}') for img in dir_imgs]
        if len(dir_lbls) > 0:
            [labels.append(f'{dir_pth}/{lbl}') for lbl in dir_lbls]
    for image, label in zip(images, labels):
        if Path(image).stem == Path(label).stem:
            image_labels[image] = label
    return images, labels, image_labels

def convert_yolo_to_ls(
    input_dir,
    out_file,
    to_name='image',
    from_name='label',
    out_type="annotations",
    image_root_url='/data/local-files/?d=',
    image_ext='.jpg,.jpeg,.png',
    yolo_type="rectanglelabels",
    image_dims: Optional[Tuple[int, int]] = None,
):
    """Convert YOLO labeling to Label Studio JSON
    :param input_dir: directory with YOLO where images, labels, notes.json are located
    :param out_file: output file with Label Studio JSON tasks
    :param to_name: object name from Label Studio labeling config
    :param from_name: control tag name from Label Studio labeling config
    :param out_type: annotation type - "annotations" or "predictions"
    :param image_root_url: root URL path where images will be hosted, e.g.: http://example.com/images
    :param image_ext: image extension/s - single string or comma separated list to search, eg. .jpeg or .jpg, .png and so on.
    :param yolo_type: label type - "rectanglelabels" or "polygonlabels"
    :param image_dims: image dimensions - optional tuple of integers specifying the image width and height of *all* images in the dataset. Defaults to opening the image to determine it's width and height, which is slower. This should only be used in the special case where you dataset has uniform image dimesions.
    """

    tasks = []
    logger.info(f'Preparing your {out_type} yolo dataset with {yolo_type} to import into LabelStudio')
    logger.info('Reading YOLO notes and categories from %s', input_dir)

    # build categories=>labels dict
    notes_file = os.path.join(input_dir, 'classes.txt')
    with open(notes_file) as f:
        lines = [line.strip() for line in f.readlines()]
    categories = {i: line for i, line in enumerate(lines)}
    logger.info(f'Found {len(categories)} categories:')
    _= [logger.info(f"\t{i}: {cat}") for i, cat in enumerate(categories.values())]


    # generate and save labeling config
    label_config_file = out_file.replace('.json', '') + '.label_config.xml'
    poly_ops = {'stroke':'3', 'pointSize':'small', 'opacity':'0.2'}
    generate_label_config(
        categories,
        {from_name: 'RectangleLabels' if yolo_type == "rectanglelabels" else 'PolygonLabels','poly_ops':poly_ops},
        to_name,
        from_name,
        label_config_file,
    )

    # define directories
    # retrieve data (image and label paths). handles datasets with data in subdirectories, e.g. train / val / test
    images, labels, image_labels = get_data(input_dir, image_ext)
    logger.info('Converting labels found recursively at %s', input_dir)
    if yolo_type == 'polygonlabels':
        # verify if current labels are boxes
        # scan labels list for first non-empty label, peek contents, determine label type
        for label in labels:
            with open(labels[0]) as f:
                sample_lbl = [line.strip() for line in f.readlines()]
            if len(sample_lbl) == 0:
                continue
            else:
                break   # non-empty label found
        logger.info(f'sample label: {sample_lbl}')
        if len(sample_lbl) < 7: # Polygons expected to consist of 7 items. At least three x,y pairs + class 
            logger.info('Your labels are bounding boxes, but you requested polygons. Transforming labels from bboxes to polygons')
            polygonise_bboxes(input_dir, labels, out_type)


    # build array out of provided comma separated image_extns (str -> array)
    image_ext = [x.strip() for x in image_ext.split(",")]
    logger.info(f'image extensions->, {image_ext}')

    # x_scale = lambda x_prop: round(x_prop*image_width,1)
    # y_scale = lambda y_prop: round((y_prop)*image_height,1)

    # formatter functions (for percent values rel to 100)
    x_scale = lambda x_prop: round(x_prop*100,2)
    y_scale = lambda y_prop: round((y_prop)*100,2)

    # loop through images
    for img in images:
        f = Path(img).stem + Path(img).suffix
        image_file_found_flag = False
        for ext in image_ext:
            if f.endswith(ext):
                image_file = f
                image_file_base = f[0 : -len(ext)]
                image_file_found_flag = True
                break
        if not image_file_found_flag:
            continue

        image_root_url += '' if image_root_url.endswith('/') else '/'
        task = {
            "data": {
                # eg. '../../foo+you.py' -> '../../foo%2Byou.py'
                "image": image_root_url
                + str(pathname2url(image_file)),
                "storage_filename": image_file
            }
        }

        # define coresponding label file and check existence
        label_file = image_labels[img]

        if os.path.exists(label_file):
            task[out_type] = [
                {
                    "result": [],
                    "ground_truth": False,
                }
            ]

            # read image sizes
            if image_dims is None:
                # default to opening file if we aren't given image dims. slow!
                with Image.open(img) as im:
                    image_width, image_height = im.size
            else:
                image_width, image_height = image_dims

            with open(label_file) as file:
                # convert all bounding boxes to Label Studio Results
                lines = file.readlines()
                
                for line in lines:
                    if yolo_type == "rectanglelabels":
                        label_id, x, y, width, height = line.split()[:5]
                        conf = line.split()[-1] if out_type == 'predictions' else None
                        x, y, width, height = (
                            float(x),
                            float(y),
                            float(width),
                            float(height),
                        )
                        conf = float(conf) if conf is not None else None
                        value = {
                            "x": (x - width / 2) * 100,
                            "y": (y - height / 2) * 100,
                            "width": width * 100,
                            "height": height * 100,
                            "rotation": 0,
                            "rectanglelabels": [categories[int(label_id)]],
                        }
                        
                    elif yolo_type == "polygonlabels":
                        parts = [float( part ) for part in line.split()]
                        label_id = int(parts.pop(0))
                        if out_type == 'predictions':          
                            conf = parts.pop(-1)
                        xy_pairs = [ [x_scale(parts[i]), y_scale(parts[i+1])] for i in range(0,len(parts),2) ] 
                        
                        value = {
                            "points": xy_pairs,
                            "polygonlabels": [categories[int(label_id)]],
                        }
                    
                    item = {
                        "id": uuid.uuid4().hex[0:10],
                        "type": yolo_type,
                        "value": value,
                        "to_name": to_name,
                        "from_name": from_name,
                        "image_rotation": 0,
                        "original_width": image_width,
                        "original_height": image_height,
                    }
                    if out_type == 'predictions':
                        item["score"] = conf
                    task[out_type][0]['result'].append(item)
                    
                        

        tasks.append(task)

    if len(tasks) > 0:
        logger.info('Saving Label Studio JSON to %s', out_file)
        with open(out_file, 'w') as out:
            json.dump(tasks, out)

        print(
            '\n'
            f'  1. Create a new project in Label Studio\n'
            f'  2. Use Labeling Config from "{label_config_file}"\n'
            f'  3. Setup serving for images [e.g. you can use Local Storage (or others):\n'
            f'     https://labelstud.io/guide/storage.html#Local-storage]\n'
            f'  4. Import "{out_file}" to the project\n'
        )
    else:
        logger.error('No labels converted')

def polygonise_bboxes(input_dir, labels, out_type):
    """
    This function allows the user to seamlessly transform existing bounding boxes 
     into polygons as they're imported into Label Studio. Ideal for datasets 
     transitioning from the yolo detect to the yolo segment task.
    :param input_dir  directory with YOLO where images, labels, notes.json are located
    """
    labels_dir = Path(input_dir) / 'labels'
    poly_labels_dir = Path(input_dir) / 'labels-seg'
    os.makedirs(poly_labels_dir, exist_ok=True)
    poly_labels = []
    for label in labels:
        # verify subrdirectory exists
        poly_label_pth = label.replace(str(labels_dir),str(poly_labels_dir))
        poly_label_subdir = Path(poly_label_pth).parent
        if not os.path.exists(poly_label_subdir):
            os.makedirs(poly_label_subdir, exist_ok=True)
        with open(label, 'r') as lbl_f:
            boxes = [line.strip() for line in lbl_f.readlines()]
        poly_boxes = []
        for box in boxes:
            c, cx, cy, w, h = [float(n) for n in box.split()[:5]]
            conf = line.split()[-1] if out_type == 'predictions' else None
            x0, y0 = (cx-(w/2), cy+(h/2))
            x1, y1 = (cx-(w/2), cy-(h/2))
            x2, y2 = (cx+(w/2), cy-(h/2))
            x3, y3 = (cx+(w/2), cy+(h/2))
            poly_boxes.append(f'{int(c)} {x0} {y0} {x1} {y1} {x2} {y2} {x3} {y3}')
            poly_boxes.append(f'{conf}\n' if out_type == 'predictions' else '\n')
        with open(poly_label_pth, 'w+') as plbl_f:
            plbl_f.write(''.join(poly_boxes))
        poly_labels.append(poly_label_pth)
    # keep copy of original bboxes labels, and make polygon labels the default one
    shutil.move( str(labels_dir),f'{str(labels_dir)}-old_boxes')
    shutil.move( str(poly_labels_dir), str(labels_dir) )

    # return poly_labels

def add_parser(subparsers):
    yolo = subparsers.add_parser('yolo')

    yolo.add_argument(
        '-i',
        '--input',
        dest='input',
        required=True,
        help='directory with YOLO where images, labels, notes.json are located',
        action=ExpandFullPath,
    )
    yolo.add_argument(
        '-o',
        '--output',
        dest='output',
        help='output file with Label Studio JSON tasks',
        default='output.json',
        action=ExpandFullPath,
    )
    yolo.add_argument(
        '--to-name',
        dest='to_name',
        help='object name from Label Studio labeling config',
        default='image',
    )
    yolo.add_argument(
        '--from-name',
        dest='from_name',
        help='control tag name from Label Studio labeling config',
        default='label',
    )
    yolo.add_argument(
        '--out-type',
        dest='out_type',
        help='annotation type - "annotations" or "predictions"',
        default='annotations',
    )
    yolo.add_argument(
        '--yolo-type',
        dest='yolo_type',
        help='label type - "rectanglelabels" or "polygonlabels" ',
        default='rectangles',
    )
    yolo.add_argument(
        '--image-root-url',
        dest='image_root_url',
        help='root URL path where images will be hosted, e.g.: http://example.com/images',
        default='/data/local-files/?d=',
    )
    yolo.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image extension to search: .jpeg or .jpg, .png',
        default='.jpg',
    )
    yolo.add_argument(
        '--image-dims',
        dest='image_dims',
        type=int,
        nargs=2,
        help=(
            "optional tuple of integers specifying the image width and height of *all* "
            "images in the dataset. Defaults to opening the image to determine it's width "
            "and height, which is slower. This should only be used in the special "
            "case where you dataset has uniform image dimesions. e.g. `--image-dims 600 800` "
            "if all your images are of dimensions width=600, height=800"
        ),
        default=None,
    )
