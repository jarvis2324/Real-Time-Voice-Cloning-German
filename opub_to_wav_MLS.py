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
already_done = []


sample_rate=16000
print("Checking subfolders...")
for speaker in speaker_dirs:
  folder_name = speaker.replace('/mnt/disks/drive_1000/Download/mls_german_opus/train','')
  if folder_name in already_done:
    print(speaker, " is already done")
    continue

  else:
    print(speaker, " is new")
    # get subfolders
    speaker_subfolders_search_dir = os.path.join(speaker, "*//")
    #print(speaker_subfolders_search_dir)
    speaker_subfolders = glob(speaker_subfolders_search_dir)
    #print(speaker_subfolders)
    for folder in speaker_subfolders:
      # if '10904' in folder or '10791' in folder or '8659' in folder or '4174' in folder:
      folder_to_be_created = folder.replace('mls_german_opus','mls_german_wav')
      #folder_to_be_created = folder.replace('/drive/MyDrive/','/')
      if os.path.exists(folder_to_be_created):
        pass

      else:
        os.makedirs(folder_to_be_created)
    #   for file in os.listdir(folder):
    #     file_path = os.path.join(folder,file)
    #     save_path = os.path.join(folder_to_be_created, file.replace('opus', 'wav'))
    #     wf = subprocess.Popen(['ffmpeg', '-loglevel', 'quiet', '-i',
    #                       file_path,
    #                       '-ar', str(sample_rate) , '-ac', '1', '-y', save_path],
    #                       stdout=subprocess.PIPE)

      # print("Before Removing")
      # total, used, free = shutil.disk_usage('/content/mls_german_opus')
      # print(total, used, free) 
      #shutil.rmtree(folder)
      # print("After Removing")
    # total, used, free = shutil.disk_usage('/content/mls_german_opus')
    # if(free < used):
    #   break
    # print(total, used, free)  