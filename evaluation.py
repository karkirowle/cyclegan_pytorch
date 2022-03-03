import soundfile
import pandas as pd
import torch
import argparse
import re
import librosa

from os.path import join, splitext, isdir, basename
from os import listdir
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text
from jiwer import wer

available_speakers = ['F02', 'F03', 'F04', 'F05', 'M01', 'M04',
                      'M05', 'M07', 'M08', 'M09', 'M10', 'M11', 'M12', 'M14', 'M16']

# from espnet_model_zoo.downloader import ModelDownloader
# from espnet2.bin.asr_inference import Speech2Text

d = ModelDownloader()
speech2text = Speech2Text(
    **d.download_and_unpack("Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best"),
    # Decoding parameters are not included in the model file
    maxlenratio=0.0,
    minlenratio=0.0,
    beam_size=20,
    ctc_weight=0.3,
    lm_weight=0.5,
    penalty=0.0,
    nbest=1
)

data_root = "/tudelft.net/staff-bulk/ewi/insy/SpeechLab/corpora/UASpeech_2"
validation_dir = "./validation_output"

available_speakers = ['F02', 'F03', 'F04', 'F05', 'M01', 'M04',
                      'M05', 'M07', 'M08', 'M09', 'M10', 'M11', 'M12', 'M14', 'M16']

word_list = pd.read_excel('./speaker_wordlist.xls',
                          sheet_name='Word_filename')


def filter_(x, validating=False):
    name_dict = re.match(
        r'(?P<speaker>[\w]+)\_(?P<rep>[\w]+)\_(?P<filename>\w+)\_(?P<mic>\w+)\.wav', x).groupdict()
    criterion = name_dict['rep'] == 'B3'
    if validating:
        return criterion
    else:
        return not criterion


def filter_1(x):
    match = re.match(
        r'(?P<speaker>[\w]+)\_(?P<rep>[\w]+)\_(?P<file_id>\w+)\_(?P<mic>\w+)\.wav', x)
    if match:
        name_dict = match.groupdict()
        criterion = name_dict['rep'] == 'B2' or (
            name_dict['rep'] == 'B1' or name_dict['rep'] == 'B3') and name_dict['file_id'].startswith('UW')
        return criterion and name_dict['mic'] == 'M3'
    else:
        return False

# def generate(files):
#     model = Generator(24).to("cuda")
#     model.load_state_dict(torch.load('model/generator_D2N'))
#     model.eval()


def collect_validation_files(dir_):
    dir = join(validation_dir, dir_)

    if not isdir(dir):
        raise RuntimeError("{} doesn't exist.".format(dir))
    files = [join(dir, f) for f in listdir(dir)]
    files = list(filter(lambda x: splitext(x)[1] == ".wav", files))
    # files = list(filter(lambda x: splitext(
    #     basename(x))[0].endswith("_M2"), files))
    files = sorted(files)

    return files


def collect_files(speaker, evaluation=False, filter_list=[]):
    """Collect wav files for specific speakers.
    Returns:
        list: List of collected wav files.
    """
    speaker_dir = join(data_root, speaker)

    if evaluation:
        speaker_dir = join(data_root, 'control', 'C' + speaker)
        filter_list = ['C' + file for file in filter_list]

    if not isdir(speaker_dir):
        raise RuntimeError("{} doesn't exist.".format(speaker_dir))
    files = [join(speaker_dir, f) for f in listdir(speaker_dir)]
    files = list(filter(lambda x: splitext(x)[1] == ".wav", files))
    files = sorted(files)
    files = list(filter(lambda x: basename(x) in filter_list, files))

    return files


def evaluate(paths):
    prediction = []
    reference = []

    for path in paths:
        filename = re.match(
            r'(?P<speaker>[\w]+)\_(?P<rep>[\w]+)\_(?P<filename>\w+)\_(?P<mic>\w+)\.wav', basename(path)).groupdict()

        word_df = word_list.loc[word_list['FILE NAME'] == filename['filename']]

        if word_df.empty:
            word_df = word_list.loc[word_list['FILE NAME'] ==
                                    filename['rep'] + '_' + filename['filename']]

        word = word_df.iloc[0]['WORD']

        wav, _ = soundfile.read(path)

        wav = librosa.util.normalize(wav)
        # noise_wav = wav[:int(rate*0.5)]
        # if self.training:

        # else:
        #     noise_wav = wav[:int(sr*0.3)]

        # wav_padded = wav_padding(wav, sr=rate, frame_period=5, multiple=4)

        # denoised_wav = nr.reduce_noise(audio_clip=wav, noise_clip=noise_wav)
        nbests = speech2text(wav)
        text, *_ = nbests[0]

        print(filename['filename'] + ':', text.lower(), '-', word.lower())

        prediction.append(text.lower())
        reference.append(word.lower())
    return prediction, reference


def wer_(prediction, reference):
    incorrect = 0

    for i, p in enumerate(prediction):
        if p != reference[i]:
            incorrect += 1

    return float(incorrect/len(prediction))


def calculate_wer(seqs_hat, seqs_true):
    """Calculate sentence-level WER score for transducer model.

    From ESPNET
    Args:
        seqs_hat (torch.Tensor): prediction (batch, seqlen)
        seqs_true (torch.Tensor): reference (batch, seqlen)

    Returns:
        (float): average sentence-level WER score

    """
    word_eds, word_ref_lens = [], []

    for i, seq_hat_text in enumerate(seqs_hat):
        seq_true_text = seqs_true[i]
        hyp_words = seq_hat_text.split()
        ref_words = seq_true_text.split()

        word_eds.append(editdistance.eval(hyp_words, ref_words))
        word_ref_lens.append(len(ref_words))

    return float(sum(word_eds)) / sum(word_ref_lens)


if __name__ == '__main__':
    print('START')
    default_cache = './preprocessed/UASpeech'
    parser = argparse.ArgumentParser(
        description="Preprocess training data.")

    parser.add_argument('--cache-dir',
                        help="cached features location", default=default_cache)
    parser.add_argument('--speakers', nargs='+',
                        help='List of speakers', choices=available_speakers)

    args = parser.parse_args()

    print(args)

    validation_paths = collect_validation_files(args.cache_dir)
    validation_files = [basename(path) for path in validation_paths]

    dysarthric_paths = []
    for speaker in args.speakers:
        dysarthric_paths.extend(collect_files(
            speaker, filter_list=validation_files))

    evaluation_paths = []
    for speaker in args.speakers:
        evaluation_paths.extend(collect_files(
            speaker, evaluation=True, filter_list=validation_files))

    print('Evaluating dysarthric speech')
    d_p, d_r = evaluate(dysarthric_paths)
    print(d_p)
    print(d_r)
    print(wer(d_r, d_p))

    print('Evaluating normal speech')
    e_p, e_r = evaluate(evaluation_paths)
    print(e_p)
    print(e_r)
    print(wer(e_r, e_p))

    # print('Evaluating corrected speech')
    # v_p, v_r = evaluate(validation_paths)
    # print(v_p)
    # print(v_r)
    # print(wer(v_r, v_p))
