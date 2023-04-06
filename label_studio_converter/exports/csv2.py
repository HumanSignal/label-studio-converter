import os
import csv
import ujson as json

from copy import deepcopy, copy

from label_studio_converter.utils import ensure_dir, get_annotator, prettify_result


def convert(item_iterator, input_data, output_dir, **kwargs):
    if str(output_dir).endswith('.csv'):
        output_file = output_dir
    else:
        ensure_dir(output_dir)
        output_file = os.path.join(output_dir, 'result.csv')

    # these keys are always presented
    keys = {'annotator', 'annotation_id', 'created_at', 'updated_at', 'lead_time'}

    # make 2 passes: the first pass is to get keys, otherwise we can't write csv without headers
    for item in item_iterator(input_data):
        record = prepare_annotation_keys(item)
        keys.update(record)

    # the second pass is to write records to csv
    with open(output_file, 'w', encoding='utf8') as outfile:
        writer = csv.DictWriter(
            outfile,
            fieldnames=list(keys),
            quoting=csv.QUOTE_NONNUMERIC,
            delimiter=kwargs['sep'],
        )
        writer.writeheader()

        for item in item_iterator(input_data):
            record = prepare_annotation(item)
            writer.writerow(record)


def prepare_annotation(item):
    record = deepcopy(item['input'])
    if item.get('id') is not None:
        record['id'] = item['id']

    for name, value in item['output'].items():
        pretty_value = prettify_result(value)
        record[name] = pretty_value

    for name, value in item['input'].items():
        if isinstance(value, dict) or isinstance(value, list):
            record[name] = json.dumps(value, ensure_ascii=False)

    record['annotator'] = get_annotator(item)
    record['annotation_id'] = item['annotation_id']
    record['created_at'] = item['created_at']
    record['updated_at'] = item['updated_at']
    record['lead_time'] = item['lead_time']

    if 'agreement' in item:
        record['agreement'] = item['agreement']

    return record


def prepare_annotation_keys(item):
    record = set(item['input'].keys())  # we don't need deepcopy for keys
    if item.get('id') is not None:
        record.add('id')

    for name, value in item['output'].items():
        record.add(name)

    for name, value in item['input'].items():
        if isinstance(value, dict) or isinstance(value, list):
            record.add(name)

    if 'agreement' in item:
        record.add('agreement')

    return record
