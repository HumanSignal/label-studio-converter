from label_studio_converter.imports.colors import COLORS


LABELS = """
  <{# TAG_NAME #} name="{# FROM_NAME #}" toName="image">
{# LABELS #}  </{# TAG_NAME #}>
"""

LABELING_CONFIG = """<View>
  <Image name="{# TO_NAME #}" value="$image"/>
{# BODY #}</View>
"""


def generate_label_config(categories, tags, to_name='image', from_name='label', filename=None):
    labels = ''
    for key in sorted(categories.keys()):
        color = COLORS[int(key) % len(COLORS)]
        label = f'    <Label value="{categories[key]}" background="rgba({color[0]}, {color[1]}, {color[2]}, 1)"/>\n'
        labels += label

    body = ''
    for from_name in tags:
        tag_body = str(LABELS) \
            .replace('{# TAG_NAME #}', tags[from_name]) \
            .replace('{# LABELS #}', labels) \
            .replace('{# TO_NAME #}', to_name) \
            .replace('{# FROM_NAME #}', from_name)
        body += f'\n  <Header value="{tags[from_name]}"/>' + tag_body

    config = str(LABELING_CONFIG).replace('{# BODY #}', body).replace('{# TO_NAME #}', to_name)

    if filename:
        with open(filename, 'w') as f:
            f.write(config)

    return config
