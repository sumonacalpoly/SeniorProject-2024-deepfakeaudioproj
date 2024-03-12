import librosa as lb
import numpy as np
import tensorflow as tf
import os

#bonafide = 1, fake = 0
def create_melspectrogram(file_path, mels, frames):
    x_train = []
    y_train = []
    for f in file_path:
        label = testing_real_or_fake(f)
        print(label)




def testing_real_or_fake(file):
    with open("LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt", 'r') as f:
        for line in f:
            fields = line.split(" ")
            if fields[1] == file.rsplit(".", 1)[0]:
                if fields[4].rstrip("\n") == "bonafide":
                    return 1
                elif fields[4].rstrip("\n") == "spoof":
                    return 0
                
def create_cnn():
    ...

def main():
    training_dir_la = "LA/LA/ASVspoof2019_LA_train/flac"
    testing_dir_la = "LA/LA/ASVspoof2019_LA_eval/flac"
    training_files_la = os.listdir(training_dir_la)
    testing_files_la = os.listdir(testing_dir_la)

    x_train, y_train = create_melspectrogram(training_files_la, 128, 128)
    


if __name__ == "__main__":
    main()
