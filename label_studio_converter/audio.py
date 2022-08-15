import os
import io
import logging
import json


from .utils import get_audio_duration, ensure_dir, download, _get_annotator


logger = logging.getLogger(__name__)


def convert_to_asr_json_manifest(input_data, output_dir, data_key, project_dir, upload_dir, download_resources):
    audio_dir_rel = 'audio'
    output_audio_dir = os.path.join(output_dir, audio_dir_rel)
    ensure_dir(output_dir), ensure_dir(output_audio_dir)
    output_file = os.path.join(output_dir, 'manifest.json')
    with io.open(output_file, mode='w') as fout:
        for item in input_data:
            audio_path = item['input'][data_key]
            try:
                audio_path = download(audio_path, output_audio_dir, project_dir=project_dir, upload_dir=upload_dir,
                                      return_relative_path=True, download_resources=download_resources)
                duration = get_audio_duration(os.path.join(output_audio_dir, os.path.basename(audio_path)))
            except:
                logger.info('Unable to download {image_path} or get audio duration. The item {item} will be skipped'.format(
                    image_path=audio_path, item=item
                ), exc_info=True)
                continue

            for texts in iter(item['output'].values()):
                if len(texts) > 0 and 'text' in texts[0]:
                    break

            transcript = texts[0]['text'][0]
            metadata = {
                'audio_filepath': audio_path,
                'duration': duration,
                'text': transcript,
                'annotator': _get_annotator(item, default='')
            }
            json.dump(metadata, fout)
            fout.write('\n')
