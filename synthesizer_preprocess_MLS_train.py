import os
from pathlib import Path

#datasets_root = '/mnt/disks/drive_1000/Download/mls_german_opus/train'
datasets_root = Path("/mnt/disks/drive_1000/Download/mls_german_wav/train/audio")
out_dir = os.path.join("/mnt/disks/drive_1000/Download/synthetiser_MLS/SV2TTS/", "/mnt/disks/drive_1000/Download/synthetiser_MLS/SV2TTS/synthesizer")

out_dir = Path(out_dir)
out_dir.mkdir(exist_ok=True, parents=True)

subfolders = ""
for file_name in os.listdir('/mnt/disks/drive_1000/Download/mls_german_wav/train/audio'):
  subfolders = subfolders + file_name + ","


from synthesizer.hparams import hparams

no_alignments = True
wav_dir = False
n_processes = None
#skip_existing = False
skip_existing = True
hparams = hparams.parse("")
datasets_name = ""


from multiprocessing.pool import Pool 
from synthesizer import audio
from functools import partial
from itertools import chain
from encoder import inference as encoder
from pathlib import Path
from utils import logmmse
from tqdm import tqdm
import numpy as np
import librosa
import os
from glob import glob


