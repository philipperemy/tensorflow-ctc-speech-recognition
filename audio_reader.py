import os
from glob import glob
from random import shuffle
from time import time

import dill
import librosa

SENTENCE_ID = 'sentence_id'
SPEAKER_ID = 'speaker_id'
FILENAME = 'filename'


def find_files(directory, pattern='**/*.wav'):
    """Recursively finds all files matching the pattern."""
    return sorted(glob(os.path.join(directory, pattern), recursive=True))


def read_audio_from_filename(filename, sample_rate):
    # import scipy.io.wavfile as wav
    # fs, audio = wav.read(filename)
    audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
    audio = audio.reshape(-1, 1)
    return audio


def extract_speaker_id(filename):
    return filename.split('/')[-2]


def extract_sentence_id(filename):
    return filename.split('/')[-1].split('_')[1].split('.')[0]


class AudioReader(object):
    def __init__(self,
                 audio_dir,
                 sample_rate,
                 cache_dir='cache',
                 speakers_sub_list=None):
        print('Initializing AudioReader()')
        print('audio_dir = {}'.format(audio_dir))
        print('cache_dir = {}'.format(cache_dir))
        print('sample_rate = {}'.format(sample_rate))
        print('speakers_sub_list = {}'.format(speakers_sub_list))
        self.audio_dir = audio_dir
        self.cache_dir = cache_dir
        self.sample_rate = sample_rate
        self.metadata = dict()  # small cache <SPEAKER_ID -> SENTENCE_ID, filename>
        self.cache = dict()  # big cache <filename, data:audio librosa, text.>

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        st = time()
        if len(find_files(cache_dir, pattern='*.pkl')) == 0:  # generate all the pickle files.
            print('Nothing found at {}. Generating all the caches now.'.format(cache_dir))
            files = find_files(audio_dir)
            assert len(files) != 0, 'Cannot find any VCTK files there. Are you sure audio_dir is correct?'
            print('Found {} files in total in {}.'.format(len(files), audio_dir))
            if speakers_sub_list is not None:
                files = list(
                    filter(lambda x: any(word in extract_speaker_id(x) for word in speakers_sub_list), files))
                print('{} files correspond to the speaker list {}.'.format(len(files), speakers_sub_list))
            assert len(files) != 0

            for filename in files:
                try:
                    target_text = open(filename.replace('wav48', 'txt').replace('.wav', '.txt'), 'r').read().strip()
                    speaker_id = extract_speaker_id(filename)
                    audio = read_audio_from_filename(filename, self.sample_rate)
                    obj = {'audio': audio,
                           'target': target_text,
                           FILENAME: filename}
                    cache_filename = filename.split('/')[-1].split('.')[0] + '_cache'
                    tmp_filename = os.path.join(cache_dir, cache_filename) + '.pkl'
                    with open(tmp_filename, 'wb') as f:
                        dill.dump(obj, f)
                        print('[DUMP AUDIO] {}'.format(tmp_filename))
                    if speaker_id not in self.metadata:
                        self.metadata[speaker_id] = {}
                    sentence_id = extract_sentence_id(filename)
                    if sentence_id not in self.metadata[speaker_id]:
                        self.metadata[speaker_id][sentence_id] = []
                    self.metadata[speaker_id][sentence_id] = {SPEAKER_ID: speaker_id,
                                                              SENTENCE_ID: sentence_id,
                                                              FILENAME: filename}
                except librosa.util.exceptions.ParameterError as e:
                    print(e)
                    print('[DUMP AUDIO ERROR SKIPPING FILENAME] {}'.format(filename))
            dill.dump(self.metadata, open(os.path.join(cache_dir, 'metadata.pkl'), 'wb'))

        print('Using the generated files at {}. Using them to load the cache. '
              'Be sure to have enough memory.'.format(cache_dir))
        self.metadata = dill.load(open(os.path.join(cache_dir, 'metadata.pkl'), 'rb'))

        pickle_files = find_files(cache_dir, pattern='*.pkl')
        for pkl_file in pickle_files:
            if 'metadata' not in pkl_file:
                with open(pkl_file, 'rb') as f:
                    obj = dill.load(f)
                    self.cache[obj[FILENAME]] = obj
        print('Cache took {0:.2f} seconds to load. {1:} keys.'.format(time() - st, len(self.cache)))

    def get_speaker_list(self):
        return sorted(list(self.metadata.keys()))

    def sample_speakers(self, speaker_list, num_speakers):
        if speaker_list is None:
            speaker_list = self.get_speaker_list()
        all_speakers = list(speaker_list)
        shuffle(all_speakers)
        speaker_list = all_speakers[0:num_speakers]
        return speaker_list
