import argparse
import os
import csv
from glob import glob
import codecs
import subprocess
import shutil


datasets_root = '/mnt/disks/drive_1000/Download/mls_german_wav/test'
wav_folders = []
print("Searching speakers...")
#speaker dir
speaker_search_dir = os.path.join(datasets_root, "audio//*")
speaker_dirs = glob(speaker_search_dir)
# print(speaker_dirs)
wav_folders = []
segment_txt = '/mnt/disks/drive_1000/Download/mls_german_opus/test/transcripts.txt'

with open(segment_txt) as f:
    lines = f.readlines()

print(lines[0])
sample_rate=48000
print("Checking subfolders...")
for speaker in speaker_dirs:
  # get subfolders
  speaker_subfolders_search_dir = os.path.join(speaker, "*//")
  #print(speaker_subfolders_search_dir)
  speaker_subfolders = glob(speaker_subfolders_search_dir)
  #print(speaker_subfolders)
  for folder in speaker_subfolders:
    print(folder)
    #folder_to_be_created = folder.replace('mls_german_opus','mls_german_wav')
    # if os.path.exists(folder_to_be_created):
    #   pass

    # else:
    #   os.makedirs(folder_to_be_created)
    for file in os.listdir(folder):
      file_path = os.path.join(folder,file)
      text_file = os.path.join(folder,file.replace('wav', 'txt'))
      #print(text_file)
      file = file.replace('.wav', '')
      #print(file)
      for line in lines:
        if file in line:
          ll = (line.replace(file,'').strip())
          with open(text_file, 'w') as f:
            f.write(ll)
          
          print(text_file, " ", ll)

      
