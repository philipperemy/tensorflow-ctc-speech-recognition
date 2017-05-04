import os
import sys

sys.path.append(os.path.abspath('..'))

from audio_reader import AudioReader
from constants import c


def generate():
    speakers_sub_list = ['p225']
    AudioReader(audio_dir=c.AUDIO.VCTK_CORPUS_PATH,
                sample_rate=c.AUDIO.SAMPLE_RATE,
                speakers_sub_list=speakers_sub_list)


if __name__ == '__main__':
    generate()
