from label_studio_converter.imports.colors import COLORS


LABELING_CONFIG = """<View>
  <Image name="{# TO_NAME #}" value="$image"/>
  <RectangleLabels name="{# FROM_NAME #}" toName="image">

{# LABELS #}
  </RectangleLabels>
</View>
"""


def generate_label_config(categories, to_name='image', from_name='label', filename=None):
    labels = ''
    for key in sorted(categories.keys()):
        color = COLORS[key % len(COLORS)]
        label = f'    <Label value="{categories[key]}" background="rgba({color[0]}, {color[1]}, {color[2]}, 1)"/>\n'
        labels += label

    config = LABELING_CONFIG \
        .replace('{# LABELS #}', labels) \
        .replace('{# TO_NAME #}', to_name) \
        .replace('{# FROM_NAME #}', from_name)

    if filename:
        with open(filename, 'w') as f:
            f.write(config)

    return config
