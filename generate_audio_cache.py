import os
from argparse import ArgumentParser

from audio_reader import AudioReader


def ensure_dir(directory):
    directory = os.path.expanduser(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def get_script_arguments():
    args = ArgumentParser()
    args.add_argument('--audio_dir', required=True)
    args.add_argument('--output_dir', default='cache', type=ensure_dir)
    args.add_argument('--sample_rate', default=16000, type=int)
    args.add_argument('--speakers_sub_list', default='p225')  # example: p225,p226,p227
    return args.parse_args()


def generate():
    args = get_script_arguments()
    if args.speakers_sub_list is None or len(args.speakers_sub_list) == 0:
        speakers_sub_list = None
    else:
        speakers_sub_list = args.speakers_sub_list.split(',')
    AudioReader(audio_dir=args.audio_dir,
                sample_rate=args.sample_rate,
                cache_dir=args.output_dir,
                speakers_sub_list=speakers_sub_list)


if __name__ == '__main__':
    generate()
