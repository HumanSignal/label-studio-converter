import os
import io
import logging
import json


from .utils import get_audio_duration, ensure_dir, download


logger = logging.getLogger(__name__)


def convert_to_asr_json_manifest(input_data, output_dir, data_key):
    audio_dir_rel = 'audio'
    output_audio_dir = os.path.join(output_dir, audio_dir_rel)
    ensure_dir(output_dir), ensure_dir(output_audio_dir)
    output_file = os.path.join(output_dir, 'manifest.json')
    with io.open(output_file, mode='w') as fout:
        for item in input_data:
            audio_path = item['input'][data_key]
            try:
                audio_path, is_downloaded = download(audio_path, output_audio_dir)
                if is_downloaded:
                    audio_filepath = os.path.join(audio_dir_rel, os.path.basename(audio_path))
                else:
                    audio_filepath = audio_path
            except:
                logger.error('Unable to download {image_path}. The item {item} will be skipped'.format(
                    image_path=audio_path, item=item
                ), exc_info=True)
            duration = get_audio_duration(audio_path)
            texts = next(iter(item['output'].values()))
            transcript = texts[0]['text'][0]
            metadata = {
                'audio_filepath': audio_filepath,
                'duration': duration,
                'text': transcript
            }
            json.dump(metadata, fout)
            fout.write('\n')
