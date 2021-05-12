import argparse
import os
import io

from label_studio_converter.converter import Format, FormatNotSupportedError
from label_studio_converter.export.csv import ExportToCSV


class ExpandFullPath(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))


def get_args():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    subparsers.required = False

    # Export converter
    parser_export = subparsers.add_parser('export')

    parser_export.add_argument(
        '-i', '--input', dest='input', required=True,
        help='Path to Label Studio export file',
        action=ExpandFullPath
    )
    parser_export.add_argument(
        '-o', '--output', dest='output',
        help='Output file or directory (will be created if not exists)',
        default=os.path.join(os.path.dirname(__file__), 'output'),
        action=ExpandFullPath
    )
    parser_export.add_argument(
        '-f', '--format', dest='format',
        metavar='FORMAT',
        help='Output format: ' + ', '.join(f.name for f in Format),
        type=Format.from_string,
        choices=list(Format),
        default=Format.JSON
    )
    parser_export.add_argument(
        '--csv-separator', dest='csv_separator',
        help='Separator used in CSV format',
        default=','
    )
    parser_export.add_argument(
        '--csv-no-header', dest='csv_no_header',
        help='Whether to omit header in CSV output file',
        action='store_true'
    )
    return parser.parse_args()


def export(args):
    if args.format == Format.CSV:
        header = not args.csv_no_header
        sep = args.csv_separator
        ExportToCSV(args.input).to_file(args.output, sep=sep, header=header, index=False)
    else:
        raise FormatNotSupportedError()


def main():
    args = get_args()
    if args.command == 'export':
        export(args)
