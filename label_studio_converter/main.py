import os
import io
import logging
import argparse

from label_studio_converter.converter import Converter, Format, FormatNotSupportedError
from label_studio_converter.exports.csv import ExportToCSV
from label_studio_converter.utils import ExpandFullPath
from label_studio_converter.imports import yolo as import_yolo, coco as import_coco

logging.basicConfig(level=logging.INFO)


def get_export_args(parser):
    parser.add_argument(
        '-i',
        '--input',
        dest='input',
        required=True,
        help='Directory or JSON file with annotations (e.g. "/<project_path>/annotations")',
        action=ExpandFullPath,
    )
    parser.add_argument(
        '-c',
        '--config',
        dest='config',
        help='Project config (e.g. "/<project_path>/config.xml")',
        action=ExpandFullPath,
    )
    parser.add_argument(
        '-o',
        '--output',
        dest='output',
        help='Output file or directory (will be created if not exists)',
        default=os.path.join(os.path.dirname(__file__), 'output'),
        action=ExpandFullPath,
    )
    parser.add_argument(
        '-f',
        '--format',
        dest='format',
        metavar='FORMAT',
        help='Output format: ' + ', '.join(f.name for f in Format),
        type=Format.from_string,
        choices=list(Format),
        default=Format.JSON,
    )
    parser.add_argument(
        '--csv-separator',
        dest='csv_separator',
        help='Separator used in CSV format',
        default=',',
    )
    parser.add_argument(
        '--csv-no-header',
        dest='csv_no_header',
        help='Whether to omit header in CSV output file',
        action='store_true',
    )
    parser.add_argument(
        '--image-dir',
        dest='image_dir',
        help='In case of image outputs (COCO, VOC, ...), specifies output image directory where downloaded images will '
        'be stored. (If not specified, local image paths left untouched)',
    )
    parser.add_argument(
        '--project-dir',
        dest='project_dir',
        default=None,
        help='Label Studio project directory path',
    )
    parser.add_argument(
        '--heartex-format',
        dest='heartex_format',
        action='store_true',
        default=True,
        help='Set this flag if your annotations are in one JSON file instead of multiple JSON files from directory',
    )


def get_all_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    subparsers.required = False

    # Export
    parser_export = subparsers.add_parser(
        'export',
        help='Converter from Label Studio JSON annotations to external formats',
    )
    get_export_args(parser_export)

    # Import
    parser_import = subparsers.add_parser(
        'import',
        help="Converter from external formats to Label Studio JSON annotations",
    )
    import_format = parser_import.add_subparsers(dest='import_format')
    import_yolo.add_parser(import_format)
    import_coco.add_parser(import_format)

    return parser.parse_args()


def export(args):
    c = Converter(args.config, project_dir=args.project_dir)

    if args.format == Format.JSON:
        c.convert_to_json(args.input, args.output)
    elif args.format == Format.CSV:
        header = not args.csv_no_header
        sep = args.csv_separator
        c.convert_to_csv(
            args.input,
            args.output,
            sep=sep,
            header=header,
            is_dir=not args.heartex_format,
        )
    elif args.format == Format.CSV_OLD:
        header = not args.csv_no_header
        sep = args.csv_separator
        ExportToCSV(args.input).to_file(
            args.output, sep=sep, header=header, index=False
        )
    elif args.format == Format.TSV:
        header = not args.csv_no_header
        sep = '\t'
        c.convert_to_csv(
            args.input,
            args.output,
            sep=sep,
            header=header,
            is_dir=not args.heartex_format,
        )
    elif args.format == Format.CONLL2003:
        c.convert_to_conll2003(args.input, args.output, is_dir=not args.heartex_format)
    elif args.format == Format.COCO:
        c.convert_to_coco(
            args.input,
            args.output,
            output_image_dir=args.image_dir,
            is_dir=not args.heartex_format,
        )
    elif args.format == Format.VOC:
        c.convert_to_voc(
            args.input,
            args.output,
            output_image_dir=args.image_dir,
            is_dir=not args.heartex_format,
        )
    elif args.format == Format.YOLO:
        c.convert_to_yolo(args.input, args.output, is_dir=not args.heartex_format)
    else:
        raise FormatNotSupportedError()


def imports(args):
    if args.import_format == 'yolo':
        import_yolo.convert_yolo_to_ls(
            input_dir=args.input,
            out_file=args.output,
            to_name=args.to_name,
            from_name=args.from_name,
            out_type=args.out_type,
            image_root_url=args.image_root_url,
            image_ext=args.image_ext,
        )

    elif args.import_format == 'coco':
        import_coco.convert_coco_to_ls(
            input_file=args.input,
            out_file=args.output,
            to_name=args.to_name,
            from_name=args.from_name,
            out_type=args.out_type,
            image_root_url=args.image_root_url,
            point_width=args.point_width,
        )
    else:
        raise FormatNotSupportedError()


def main():
    args = get_all_args()
    if args.command == 'export':
        export(args)
    elif args.command == 'import':
        imports(args)
    else:
        print('Please, use "import" or "export" or "-h" command')


if __name__ == "__main__":
    main()
