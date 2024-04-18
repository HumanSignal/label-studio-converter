# Label Studio Converter

[Website](https://labelstud.io/) • [Docs](https://labelstud.io/guide) • [Twitter](https://twitter.com/heartexlabs) • [Join Slack Community <img src="https://app.heartex.ai/docs/images/slack-mini.png" width="18px"/>](https://slack.labelstud.io)

## Table of Contents

- [Introduction](#introduction)
- [Examples](#examples)
    - [JSON](#json)
    - [CSV](#csv)
    - [CoNLL 2003](#conll-2003)
    - [COCO](#coco)
    - [Pascal VOC XML](#pascal-voc-xml)
- [YOLO to Label Studio Converter](#yolo-to-label-studio-converter)
  - [Usage](#usage)
  - [Tutorial: Importing YOLO Pre-Annotated Images to Label Studio using Local Storage](#tutorial-importing-yolo-pre-annotated-images-to-label-studio-using-local-storage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Label Studio Format Converter helps you to encode labels into the format of your favorite machine learning library.

## Examples

#### JSON
**Running from the command line:**

```bash
pip install -U label-studio-converter
python label-studio-converter export -i exported_tasks.json -c examples/sentiment_analysis/config.xml -o output_dir -f CSV
```

**Running from python:**
```python
from label_studio_converter import Converter

c = Converter('examples/sentiment_analysis/config.xml')
c.convert_to_json('examples/sentiment_analysis/completions/', 'tmp/output.json')
```

Getting output file: `tmp/output.json`
```json
[
  {
    "reviewText": "Good case, Excellent value.",
    "sentiment": "Positive"
  },
  {
    "reviewText": "What a waste of money and time!",
    "sentiment": "Negative"
  },
  {
    "reviewText": "The goose neck needs a little coaxing",
    "sentiment": "Neutral"
  }
]
```

Use cases: any tasks


#### CSV
Running from the command line:
```bash
python label_studio_converter/cli.py --input examples/sentiment_analysis/completions/ --config examples/sentiment_analysis/config.xml --output output_dir --format CSV --csv-separator $'\t'
```

Running from python:
```python
from label_studio_converter import Converter

c = Converter('examples/sentiment_analysis/config.xml')
c.convert_to_csv('examples/sentiment_analysis/completions/', 'output_dir', sep='\t', header=True)
```

Getting output file `tmp/output.tsv`:
```tsv
reviewText	sentiment
Good case, Excellent value.	Positive
What a waste of money and time!	Negative
The goose neck needs a little coaxing	Neutral
```

Use cases: any tasks

#### CoNLL 2003

Running from the command line:
```bash
python label_studio_converter/cli.py --input examples/named_entity/completions/ --config examples/named_entity/config.xml --output tmp/output.conll --format CONLL2003
```

Running from python:
```python
from label_studio_converter import Converter

c = Converter('examples/named_entity/config.xml')
c.convert_to_conll2003('examples/named_entity/completions/', 'tmp/output.conll')
```

Getting output file `tmp/output.conll`
```text
-DOCSTART- -X- O
Showers -X- _ O
continued -X- _ O
throughout -X- _ O
the -X- _ O
week -X- _ O
in -X- _ O
the -X- _ O
Bahia -X- _ B-Location
cocoa -X- _ O
zone, -X- _ O
...
```

Use cases: text tagging


#### COCO
Running from the command line:
```bash
python label_studio_converter/cli.py --input examples/image_bbox/completions/ --config examples/image_bbox/config.xml --output tmp/output.json --format COCO --image-dir tmp/images
```

Running from python:
```python
from label_studio_converter import Converter

c = Converter('examples/image_bbox/config.xml')
c.convert_to_coco('examples/image_bbox/completions/', 'tmp/output.conll', output_image_dir='tmp/images')
```

Output images could be found in `tmp/images`

Getting output file `tmp/output.json`
```json
{
  "images": [
    {
      "width": 800,
      "height": 501,
      "id": 0,
      "file_name": "tmp/images/62a623a0d3cef27a51d3689865e7b08a"
    }
  ],
  "categories": [
    {
      "id": 0,
      "name": "Planet"
    },
    {
      "id": 1,
      "name": "Moonwalker"
    }
  ],
  "annotations": [
    {
      "id": 0,
      "image_id": 0,
      "category_id": 0,
      "segmentation": [],
      "bbox": [
        299,
        6,
        377,
        260
      ],
      "ignore": 0,
      "iscrowd": 0,
      "area": 98020
    },
    {
      "id": 1,
      "image_id": 0,
      "category_id": 1,
      "segmentation": [],
      "bbox": [
        288,
        300,
        132,
        90
      ],
      "ignore": 0,
      "iscrowd": 0,
      "area": 11880
    }
  ],
  "info": {
    "year": 2019,
    "version": "1.0",
    "contributor": "Label Studio"
  }
}
```

Use cases: image object detection

#### Pascal VOC XML
Running from the command line:
```bash
python label_studio_converter/cli.py --input examples/image_bbox/completions/ --config examples/image_bbox/config.xml --output tmp/voc-annotations --format VOC --image-dir tmp/images
```

Running from python:
```python
from label_studio_converter import Converter

c = Converter('examples/image_bbox/config.xml')
c.convert_to_voc('examples/image_bbox/completions/', 'tmp/output.conll', output_image_dir='tmp/images')
```

Output images can be found in `tmp/images`

Corresponding annotations could be found in `tmp/voc-annotations/*.xml`:
```xml
<?xml version="1.0" encoding="utf-8"?>
<annotation>
<folder>tmp/images</folder>
<filename>62a623a0d3cef27a51d3689865e7b08a</filename>
<source>
<database>MyDatabase</database>
<annotation>COCO2017</annotation>
<image>flickr</image>
<flickrid>NULL</flickrid>
</source>
<owner>
<flickrid>NULL</flickrid>
<name>Label Studio</name>
</owner>
<size>
<width>800</width>
<height>501</height>
<depth>3</depth>
</size>
<segmented>0</segmented>
<object>
<name>Planet</name>
<pose>Unspecified</pose>
<truncated>0</truncated>
<difficult>0</difficult>
<bndbox>
<xmin>299</xmin>
<ymin>6</ymin>
<xmax>676</xmax>
<ymax>266</ymax>
</bndbox>
</object>
<object>
<name>Moonwalker</name>
<pose>Unspecified</pose>
<truncated>0</truncated>
<difficult>0</difficult>
<bndbox>
<xmin>288</xmin>
<ymin>300</ymin>
<xmax>420</xmax>
<ymax>390</ymax>
</bndbox>
</object>
</annotation>
```

Use cases: image object detection

--------

# YOLO to Label Studio Converter 

### YOLO directory structure

Check the structure of YOLO folder first, keep in mind that the root is `/yolo/datasets/one`. 

```
/yolo/datasets/one
  images
   - 1.jpg
   - 2.jpg
   - ...
  labels
   - 1.txt
   - 2.txt

  classes.txt
```

*classes.txt example*

```
Airplane
Car
```

### Usage

```
label-studio-converter import yolo -i /yolo/datasets/one -o ls-tasks.json --image-root-url "/data/local-files/?d=one/images"
```
Where the URL path from `?d=` is relative to the path you set in `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT`.

**Note for Local Storages** 
  * It's very important to set `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/yolo/datasets` (**not** to `/yolo/datasets/one`, but **`/yolo/datasets`**) for Label Studio to run.
  * [Add a new Local Storage](https://labelstud.io/guide/storage#Local-storage) in the project settings and set **Absolute local path** to `/yolo/datasets/one/images` (or `c:\yolo\datasets\one\images` for Windows). 

**Note for Cloud Storages**
  * Use `--image-root-url` to make correct prefixes for task URLs, e.g. `--image-root-url s3://my-bucket/yolo/datasets/one`.
  * [Add a new Cloud Storage](https://labelstud.io/guide/storage) in the project settings with the corresponding bucket and prefix.

**Help command**

```
label-studio-converter import yolo -h

usage: label-studio-converter import yolo [-h] -i INPUT [-o OUTPUT]
                                          [--to-name TO_NAME]
                                          [--from-name FROM_NAME]
                                          [--out-type OUT_TYPE]
                                          [--image-root-url IMAGE_ROOT_URL]
                                          [--image-ext IMAGE_EXT]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        directory with YOLO where images, labels, notes.json
                        are located
  -o OUTPUT, --output OUTPUT
                        output file with Label Studio JSON tasks
  --to-name TO_NAME     object name from Label Studio labeling config
  --from-name FROM_NAME
                        control tag name from Label Studio labeling config
  --out-type OUT_TYPE   annotation type - "annotations" or "predictions"
  --image-root-url IMAGE_ROOT_URL
                        root URL path where images will be hosted, e.g.:
                        http://example.com/images or s3://my-bucket
  --image-ext IMAGE_EXT
                        image extension to search: .jpg, .png
```


## Tutorial: Importing YOLO Pre-Annotated Images to Label Studio using Local Storage

This tutorial will guide you through the process of importing a folder with YOLO annotations into Label Studio for further annotation. 
We'll cover setting up your environment, converting YOLO annotations to Label Studio's format, and importing them into your project.


### Prerequisites
- Label Studio installed locally
- YOLO annotated images and corresponding .txt label files in the directory `/yolo/datasets/one`.
- label-studio-converter installed (available via `pip install label-studio-converter`)

### Step 1: Set Up Your Environment and Run Label Studio
Before starting Label Studio, set the following environment variables to enable Local Storage file serving:

Unix systems: 
```
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/yolo/datasets
label-studio
```

Windows: 
```
set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=C:\\yolo\\datasets
label-studio
```

Replace `/yolo/datasets` with the actual path to your YOLO datasets directory.

### Step 2: Setup Local Storage
1. Create a new project.
2. Go to the project settings and select **Cloud Storage**. 
3. Click **Add Source Storage** and select **Local files** from the **Storage Type** options.  
3. Set the **Absolute local path** to `/yolo/datasets/one/images` or `c:\yolo\datasets\one\images` on Windows.
4. Click `Add storage`.

Check more details about Local Storages [in the documentation](https://labelstud.io/guide/storage.html#Local-storage).

### Step 3: Verify Image Access
Before importing the converted annotations from YOLO, verify that you can access an image from your Local storage via Label Studio. Open a new browser tab and enter the following URL:

```
http://localhost:8080/data/local-files/?d=one/images/<your_image>.jpg
```

Replace `one/images/<your_image>.jpg` with the path to one of your images. The image should display **in the new tab of the browser**.
If you can't open an image, the Local Storage configuration is incorrect. The most likely reason is that you made a mistake when specifying your `Path` in Local Storage settings or in `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT`. 

**Note:** The URL path from `?d=` should be relative to `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/yolo/datasets`, 
it means that the real path will be `/yolo/datasets/one/images/<your_image>.jpg` and this image should exist on your hard drive.

### Step 4: Convert YOLO Annotations
Use the label-studio-converter to convert your YOLO annotations to a format that Label Studio can understand:

```
label-studio-converter import yolo -i /yolo/datasets/one -o output.json --image-root-url "/data/local-files/?d=one/images"
```

### Step 5: Import Converted Annotations
Now import the `output.json` file into Label Studio:
1. Go to your Label Studio project.
2. From the Data Manager, click **Import**.
3. Select the `output.json` file and import it.

### Step 6: Verify Annotations
After importing, you should see your images with the pre-annotated bounding boxes in Label Studio. Verify that the annotations are correct and make any necessary adjustments.

### Troubleshooting
If you encounter issues with paths or image access, ensure that:
- The LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT is set correctly.
- The `--image-root-url` in the conversion command matches the relative path:
```
`Absolute local path from Local Storage Settings` - `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT` = `path for --image_root_url`
```
e.g.:
```
/yolo/datasets/one/images - /yolo/datasets/ = one/images
```
- The Local Storage in Label Studio is set up correctly with the Absolute local path to your images (`/yolo/datasets/one/images`)
- For more details, refer to the documentation on [importing pre-annotated data](https://labelstud.io/guide/predictions.html) and [setting up Cloud Storages](https://labelstud.io/guide/storage).

------------

# Contributing

We would love to get your help for creating converters to other models. Please feel free to create pull requests.

- [Contributing Guideline](https://github.com/heartexlabs/label-studio/blob/develop/CONTRIBUTING.md)
- [Code Of Conduct](https://github.com/heartexlabs/label-studio/blob/develop/CODE_OF_CONDUCT.md)

# License

This software is licensed under the [Apache 2.0 LICENSE](/LICENSE) © [Heartex](https://www.heartex.com/). 2020

<img src="https://github.com/heartexlabs/label-studio/blob/master/images/opossum_looking.png?raw=true" title="Hey everyone!" height="140" width="140" />
