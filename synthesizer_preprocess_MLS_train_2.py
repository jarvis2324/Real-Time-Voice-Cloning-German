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


def process_utterance(wav: np.ndarray, text: str, out_dir: Path, basename: str, 
                      skip_existing: bool, hparams):
    ## FOR REFERENCE:
    # For you not to lose your head if you ever wish to change things here or implement your own
    # synthesizer.
    # - Both the audios and the mel spectrograms are saved as numpy arrays
    # - There is no processing done to the audios that will be saved to disk beyond volume  
    #   normalization (in split_on_silences)
    # - However, pre-emphasis is applied to the audios before computing the mel spectrogram. This
    #   is why we re-apply it on the audio on the side of the vocoder.
    # - Librosa pads the waveform before computing the mel spectrogram. Here, the waveform is saved
    #   without extra padding. This means that you won't have an exact relation between the length
    #   of the wav and of the mel spectrogram. See the vocoder data loader.
    
    
    # Skip existing utterances if needed
    mel_fpath = out_dir.joinpath("mels", "mel-%s.npy" % basename)
    wav_fpath = out_dir.joinpath("audio", "audio-%s.npy" % basename)
    #if skip_existing and mel_fpath.exists() and wav_fpath.exists():
    #     return None
    #print(skip_existing)
    #if skip_existing and wav_fpath.exists() and ('7958' not in str(wav_fpath)):
    #if skip_existing and wav_fpath.exists():
    if skip_existing and mel_fpath.exists() and wav_fpath.exists():
      #print("Skipping", wav_fpath )
    #   with open('/content/file_present.txt','a') as f:
    #     f.write("Skipping" + str(wav_fpath) + " mel" +  str(mel_fpath))
    #     f.write("\n")
      return None
    # if str(mel_fpath) in list_of_mel_files_written and str(wav_fpath) in list_of_audio_files_written:
    #   print("Skipping Wav ", wav_fpath, "Mel ", mel_fpath)
    #   return None

    # Trim silence
    if hparams.trim_silence:
        wav = encoder.preprocess_wav(wav, normalize=False, trim_silence=True)				  
    
    # Skip utterances that are too short
    if len(wav) < hparams.utterance_min_duration * hparams.sample_rate:
        return None
    
    # Compute the mel spectrogram
    mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
    mel_frames = mel_spectrogram.shape[1]
    
    # Skip utterances that are too long
    if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
        return None
    
    # Write the spectrogram, embed and audio to disk
    np.save(mel_fpath, mel_spectrogram.T, allow_pickle=False)
    np.save(wav_fpath, wav, allow_pickle=False)
    print("Saved File", mel_fpath, " ", wav_fpath)
    # with open('/content/file_written.txt','a') as f:
    #   f.write("writing" + str(wav_fpath) + " mel" +  str(mel_fpath))
    #   f.write("\n")

    # Return a tuple describing this training example
    return wav_fpath.name, mel_fpath.name, "embed-%s.npy" % basename, len(wav), mel_frames, text
 
 
def embed_utterance(fpaths, encoder_model_fpath):
    if not encoder.is_loaded():
        encoder.load_model(encoder_model_fpath)

    # Compute the speaker embedding of the utterance
    wav_fpath, embed_fpath = fpaths
    wav = np.load(wav_fpath)
    wav = encoder.preprocess_wav(wav)
    embed = encoder.embed_utterance(wav)
    np.save(embed_fpath, embed, allow_pickle=False)


def preprocess_dataset(datasets_root: Path, out_dir: Path, n_processes: int,
                           skip_existing: bool, hparams, no_alignments: bool,
                           datasets_name: str, subfolders: str, wav_dir: bool):
    #print("n_processes", n_processes)
    
    # Gather the input directories
    dataset_root = datasets_root.joinpath(datasets_name)
    input_dirs = [dataset_root.joinpath(subfolder.strip()) for subfolder in subfolders.split(",")]
    #print("\n    ".join(map(str, ["Using data from:"] + input_dirs)))
    assert all(input_dir.exists() for input_dir in input_dirs)
    
    # Create the output directories for each output file type
    out_dir.joinpath("mels").mkdir(exist_ok=True)
    out_dir.joinpath("audio").mkdir(exist_ok=True)
    
    # Create a metadata file
    metadata_fpath = out_dir.joinpath("train.txt")
    metadata_file = metadata_fpath.open("a" if skip_existing else "w", encoding="utf-8")

    # Preprocess the dataset
    speaker_dirs = list(chain.from_iterable(input_dir.glob("*") for input_dir in input_dirs))
    print("Speaker dirs: " + str(speaker_dirs))
    func = partial(preprocess_speaker, out_dir=out_dir, skip_existing=skip_existing, 
                   hparams=hparams, no_alignments=no_alignments, wav_dir=wav_dir)
    #n_processes = 8
    job = Pool(n_processes).imap(func, speaker_dirs)
    #print("Job is", job)
    for speaker_metadata in tqdm(job, datasets_name, len(speaker_dirs), unit="speakers"):
        print("Speaker metadata: " + str(speaker_metadata))
        for metadatum in speaker_metadata:
            metadata_file.write("|".join(str(x) for x in metadatum) + "\n")
    metadata_file.close()

    # Verify the contents of the metadata file
    print("Medata file: " + str(metadata_fpath))
    with metadata_fpath.open("r", encoding="utf-8") as metadata_file:
        metadata = [line.split("|") for line in metadata_file]
    mel_frames = sum([int(m[4]) for m in metadata])
    timesteps = sum([int(m[3]) for m in metadata])
    sample_rate = hparams.sample_rate
    hours = (timesteps / sample_rate) / 3600
    print("The dataset consists of %d utterances, %d mel frames, %d audio timesteps (%.2f hours)." %
          (len(metadata), mel_frames, timesteps, hours))
    print("Max input length (text chars): %d" % max(len(m[5]) for m in metadata))
    print("Max mel frames length: %d" % max(int(m[4]) for m in metadata))
    print("Max audio timesteps length: %d" % max(int(m[3]) for m in metadata))


def preprocess_speaker(speaker_dir, out_dir: Path, skip_existing: bool, hparams, no_alignments: bool, wav_dir: False):
    metadata = []
    #print("Speaker Dir in Preprocess",speaker_dir)
    #print("Speaker Dir in Preprocess",speaker_dir.glob('*.wav'))
    #for book_dir in speaker_dir.glob("*\\"):
    speaker_dir_temp = str(speaker_dir)
    #print("Speaker Dir is",speaker_dir_temp)
    for book_dir in speaker_dir.glob("*"):
    #print("Speaker Dir",type(speaker_dir))
    #for book_dir in os.path.listdir(speaker_dir_2):
        #print("Book Dir", book_dir)
        # if not os.path.isdir(book_dir):
        #     print("" + str(book_dir) + " is not a directory.")
        #     continue
        # print("Book dir: " + str(book_dir))
        #print(book_dir)
        if '.wav' in book_dir:
                

            wav_fpath = book_dir
            #print("Wav path: " + str(wav_fpath))
            # Load the audio waveform
            # wav, _ = librosa.load(str(wav_fpath), hparams.sample_rate)
            # if hparams.rescale:
            #     wav = wav / np.abs(wav).max() * hparams.rescaling_max

            # Get the corresponding text
            # Check for .txt (for compatibility with other datasets)
            text_fpath = wav_fpath.with_suffix(".txt")
            print(text_fpath)
            #print("Text Fpath", text_fpath)
    #         if not text_fpath.exists():
    #             print("Text_fpath does not exist: " + str(text_fpath))
    #             # Check for .normalized.txt (LibriTTS)
    #             text_fpath = wav_fpath.with_suffix(".normalized.txt")
    #             #assert text_fpath.exists()
    #             if not text_fpath.exists():
    #                 continue
    #         with text_fpath.open("r", encoding="utf-8") as text_file:
    #             text = "".join([line for line in text_file])
    #             text = text.replace("\"", "")
    #             text = text.strip()

    #         #print("Text is", text)
    #         # Process the utterance
    #         metadata.append(process_utterance(wav, text, out_dir, str(wav_fpath.with_suffix("").name),
    #                                             skip_existing, hparams))
    #     #print(metadata)
    # return [m for m in metadata if m is not None]

preprocess_dataset(datasets_root,out_dir,n_processes,skip_existing,hparams,no_alignments,datasets_name,subfolders,wav_dir)
