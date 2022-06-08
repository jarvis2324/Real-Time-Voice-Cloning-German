from pydub import AudioSegment
import os

def trim(filename, timespan, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    t1 = 0
    t2 = timespan
    t1 = t1 * 1000 #Works in milliseconds
    t2 = timespan * 1000
    k = 0
    newAudio = AudioSegment.from_file(filename)
    # wav_filename = filename.split('.')[0] + '.wav'
    # wav_folder = output_folder + '/' + wav_filename
    # file_handle = track.export(wav_folder, format='wav')
    # newAudio = AudioSegment.from_wav(wav_folder)
    length = newAudio.duration_seconds
    length = length * 1000
    print(length)
    while(t2 < length):
        output_file = output_folder + "/" + str(k) + ".wav"
        print("Saving file: " + output_file)
        print("From: " + str(t1) + " to: " + str(t2))
        output = newAudio[t1:t2]
        print(output.duration_seconds)
        output.export(output_file, format="wav") #Exports to a wav file in the current path.
        t1 = t1 + (timespan * 1000)
        t2 = t2 + (timespan * 1000)
        k = k + 1

if __name__ == '__main__':
    filename = './female_german_audio/Neue Aufnahme 12.m4a'
    output_folder = './female_output'
    timespan = 6
    trim(filename, timespan, output_folder)

