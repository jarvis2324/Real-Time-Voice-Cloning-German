import argparse
import os
import csv
from glob import glob
import codecs
import subprocess
import shutil

datasets_root = '/mnt/disks/drive_1000/Download/mls_german_opus/train'
wav_folders = []

segment_txt = '/mnt/disks/drive_1000/Download/mls_german_opus/train/transcripts.txt'

with open(segment_txt) as f:
    lines = f.readlines()

print(lines[0])
	

