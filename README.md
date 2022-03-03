# CycleGAN-VC PyTorch

This is a reimplementation of CycleGAN-VC in PyTorch written by Bence Halpern.
Additional work on UASpeech enhancement was done by Luke Prananta.

Code is available under GPL-3 license

# Installation

Main requirements:
- Python 3.5
- nnmnkwii 0.0.20
- PyTorch 1.5.0
- librosa 0.7.2
- PyWorld 0.2.10

To create the environment use the following command:

```bash
conda env create -f environment.yml
```

Downloading the VCC2016 dataset is done by first running the following command:

```bash
python download.py
```

The model can be trained with the following command:
```bash
python train.py
```


Conversion of the utterances can be done through the convert.py file, more detail on the help



```bash

Convert CycleGAN utterance

optional arguments:
  -h, --help            show this help message and exit
  --file_path FILE_PATH
                        Path of speech file to convert.
  --output_dir OUTPUT_DIR
                        Output directory for converted voice.
  --data_root DATA_ROOT
                        VCC 2016 dataroot
  --domain_A            Check if converting from domain A


```
## Short summary of what each file does

```bash
├── analysis.py - ?
├── clean_uaspeech.py - Generates the "cleaned" version of the UA Speech dataset (assumes data is denoised)
├── convert.py - Convert CycleGAN utterance
├── cyclegan.py - CycleGAN model
├── data_utils.py - Data utils 
├── download.py - Downloads the VCC2020 dataset
├── environment.yml - Conda environment file (TODO: Needs to be checked)
├── evaluation.py - 
├── f0_wrapper.py - 
├── mcep_wrapper.py -
├── modules.py  - 
├── NOTES.MD - Notes that Bence has written down during the implementation
├── preprocess.py - Preprocesses the VCC2020 dataset?
├── preprocess_test.py - 
├── preprocess_ts.py - 
├── README.md
├── requirements.txt
├── speaker_wordlist.xls
├── speedup.py
├── test.py
├── train_with_test.py
├── uaspeech.py
├── utils.py
```

### How do I retrain it with my own data?

The current way of training is through nnmnkwii FileDataSource wrapper, see the data_utils.py for an example.
This can be surprisingly efficient, because many standard speech datasets have a nanamin wrapper, so it essentially
requires you to inherit the right dataset class and provide the right dataset path. 

You can also implement custom wrapper for your own datasets. Please have a thorough look at [relevant documentation of
nnmnkwii here](https://r9y9.github.io/nnmnkwii/stable/references/datasets.html).

I might implement friendlier support for custom retraining in the future.

### Do you have a pretrained model?

Yes, of course. You can download it from my Google Drive [folder](https://drive.google.com/drive/folders/17pZhPlLDfn_wjJZLfTGKqf9lREHyHNm4?usp=sharing).
Make a folder called checkpoint and put these there.


 

 