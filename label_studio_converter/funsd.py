""" This code allows to export Label Studio Export JSON to FUNSD format. 
It's only the basic converter, it converts every bbox as a separate word. 

Usage: funsd.py export.json 
This command will export your LS OCR annotations to "./funsd/" directory. 
"""
import os
import json
from collections import defaultdict


def convert_annotation_to_fund(result):
    # collect all LS results and combine labels, text, coordinates into one record
    pre = defaultdict(dict)
    for item in result:
        o = pre[item['id']]

        labels = item.get('value', {}).get('labels', None)
        if labels:
            o['label'] = labels[0]

        text = item.get('value', {}).get('text', None)
        if text:
            o['text'] = text[0]

        if 'box' not in o:
            w, h = item['original_width'], item['original_height']
            v = item.get('value')
            x1 = v['x'] / 100.0 * w
            y1 = v['y'] / 100.0 * h
            x2 = x1 + v['width'] / 100.0 * w
            y2 = y1 + v['height'] / 100.0 * h
            o['box'] = [x1, x2, y1, y2]

    # make FUNSD output
    output = []
    counter = 0
    for key in pre:
        counter += 1
        output.append({
            "id": counter,
            "box": pre[key]['box'],
            "text": pre[key]['text'],
            "label": pre[key]['label'],
            "words": [
                {
                    "box": pre[key]['box'],
                    "text": pre[key]['text']
                }
            ],
            "linking": []
        })

    return {'form': output}


def ls_to_funsd_converter(ls_export_path='export.json', funsd_dir='funsd', data_key='ocr'):
    with open(ls_export_path) as f:
        tasks = json.load(f)

    os.makedirs(funsd_dir, exist_ok=True)

    for task in tasks:
        for annotation in task['annotations']:
            output = convert_annotation_to_fund(annotation['result'])
            filename = task["data"][data_key]
            filename = os.path.basename(filename)
            filename = f'{funsd_dir}/task-{task["id"]}-annotation-{annotation["id"]}-' \
                       f'{filename}.json'

            with open(filename, 'w') as f:
                json.dump(output, f)


if __name__ == '__main__':
	import sys
	print('Usage:', sys.argv[0], 'export.json')
	print('This command will export your LS OCR annotations to "./funsd/" directory')
	
    ls_to_funsd_converter(sys.argv[1])
