""" PathTrack BBoxes to Label Studio convert
https://www.trace.ethz.ch/publications/2017/pathtrack/index.html
"""
import os
import json
import uuid
import logging

from types import SimpleNamespace

logger = logging.getLogger('root')

try:
    import bs4
except ImportError:
    logger.info('To use PathTrack convert do "pip install bs4"')


def get_labels():
    return {}


def get_info(path):
    with open(path) as f:
        b = bs4.BeautifulSoup(f.read())
        ratio = 24 / float(b.root.doc.fps.get_text())  # todo: remove it
        return SimpleNamespace(
            fps=24,  # float(b.root.doc.fps.get_text()),  # todo: back!
            original_width=int(b.root.doc.imw.get_text()),
            original_height=int(b.root.doc.imh.get_text()),
            frame_count=int(int(b.root.doc.num_frames.get_text()) * ratio)  # todo: remove ratio
        )


def new_task(data, result, ground_truth=False, model_version=None, score=None):
    """
    :param data: dict
    :param result: annotation results with regions
    :param ground_truth: only for annotations
    :param model_version: if not None, there will be prediction
    :param score: prediction score, used only if model_version is not None
    """
    task = {
        "data": data
    }

    # add predictions or annotations
    if model_version:
        task["predictions"] = [{
            "result": result,
            "model_version": model_version,
            "score": score
        }]
    else:
        task["annotations"] = [{
            "result": result,
            "ground_truth": ground_truth
        }]

    return task


def new_region(labels, info, from_name, to_name):
    region = {
        "id": uuid.uuid4().hex[0:10],
        "type": "videorectangle",
        "value": {
            "sequence": [

            ],
            "framesCount": info.frame_count
        },
        "origin": "manual",
        "to_name": to_name,
        "from_name": from_name
    }

    if labels is not None:
        region['value']['labels'] = labels

    return region


def new_keyframe(region, bbox, info):
    region['value']['sequence'].append(
        {
            "x": bbox.x / info.original_width * 100,
            "y": bbox.y / info.original_height * 100,
            "width": bbox.width / info.original_width * 100,
            "height": bbox.height / info.original_height * 100,
            "time": bbox.frame / info.fps,
            "frame": bbox.frame,
            "enabled": False,
            "rotation": 0
        }
    )
    return region


def generator(path):
    with open(path) as f:
        lines = f.readlines()

    for line in lines:
        v = line.split()
        yield SimpleNamespace(
            frame=int(v[0]),
            bbox_id=int(v[1]),
            label_id=int(v[1]),

            x=int(v[2]), y=int(v[3]),
            width=int(v[4]), height=int(v[5])
        )


def create_config(from_name='box', to_name='video', source_value='video'):
    return f"""<View>
   <Header>Label the video:</Header>
   <Video name="{to_name}" value="${source_value}" framerate="24"/>
   <VideoRectangle name="{from_name}" toName="{to_name}" />
   <Labels name="videoLabels" toName="{to_name}" allowEmpty="true">
     <Label value="Man" background="green"/>
     <Label value="Woman" background="blue"/>
     <Label value="Other" background="blue"/>
   </Labels>
</View>
    """


def convert_shot(input_url, label_file, info_file,
                 from_name='box', to_name='video', source_value='video'):
    """ Convert bounding boxes from PathTrack to Label Studio video format

    :param input_url: video file
    :param label_file: text file with annotations line by line
    :param info_file: info.xml with frame rate and other useful info
    :param from_name: control tag name from Label Studio labeling config
    :param to_name: object name from Label Studio labeling config
    :param source_value: source name for Video tag, e.g. $video
    """
    logger.info('Convert shot start: %s', input_url)
    info = get_info(info_file)
    label_map = get_labels()  # todo: implement get_labels()
    regions = {}
    keyframe_count = 0

    # convert all bounding boxes to Label Studio Results
    for v in generator(label_file):
        idx = v.label_id
        if idx in regions:
            region = regions[idx]
        else:
            labels = [label_map[idx]] if idx in label_map else None
            region = regions[idx] = new_region(labels, info, from_name, to_name)

        # enable previous lifespan
        if len(regions[idx]['value']['sequence']) > 0:
            regions[idx]['value']['sequence'][-1]['enabled'] = True

        regions[idx] = new_keyframe(region, bbox=v, info=info)
        keyframe_count += 1

    logger.info('Shot with %i regions and %i keyframes created', len(regions), keyframe_count)
    data = {source_value: input_url}
    return new_task(data, result=list(regions.values()))


def convert_dataset(root_dir, root_url, from_name='box', to_name='video', source_value='video'):
    """ Convert PathTrack dataset to Label Studio video labeling format

    :param root_dir: root dir with video folders, e.g.: 'pathtrack/train' or 'pathtrack/test'
    :param root_url: where the dataset is served by http/https
    :param to_name: object name from Label Studio labeling config
    :param from_name: control tag name from Label Studio labeling config
    :param source_value: source name for Video tag, e.g. $video
    """
    logger.info('Convert dataset start: %s', root_dir)
    tasks = []

    if not root_url.endswith('/'):
        root_url += '/'

    for d in os.listdir(root_dir):
        shot_dir = os.path.join(root_dir, d)
        if not os.path.isdir(shot_dir):
            continue

        input_url = root_url + d + '/video.mp4'
        label_file = os.path.join(shot_dir, 'gt/gt.txt')
        info_file = os.path.join(shot_dir, 'info.xml')

        tasks.append(
            convert_shot(input_url, label_file, info_file, from_name, to_name, source_value)
        )

    path = os.path.join(root_dir, 'import.json')
    logger.info('Saving Label Studio JSON: %s', path)
    with open(path, 'w') as f:
        json.dump(tasks, f)

    path = os.path.join(root_dir, 'config.xml')
    logger.info('Saving Labeling Config: %s', path)
    config = create_config(from_name, to_name, source_value)
    with open(path, 'w') as f:
        f.write(config)


if __name__ == '__main__':
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else 'https://data.heartex.net/pathtrack/train/'
    convert_dataset('./', url)

    # convert_dataset('../../tests', 'https://data.heartex.net/pathtrack/train/')
