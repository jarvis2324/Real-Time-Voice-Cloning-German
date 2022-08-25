import argparse
import os
import csv
from glob import glob
import codecs
import subprocess
import shutil
datasets_root = '/mnt/disks/drive_1000/Download/mls_german_opus/train'
wav_folders = []
print("Searching speakers...")
#speaker dir
speaker_search_dir = os.path.join(datasets_root, "audio//*")
speaker_dirs = glob(speaker_search_dir)
print(speaker_dirs)
