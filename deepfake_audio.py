import librosa as lb
import numpy as np
import tensorflow as tf
import os

#bonafide = 1, fake = 0
def create_melspectrogram(file_name, mels, frames):
    x_train = []
    y_train = []
    for f in file_name:
        label = testing_real_or_fake(f)
        #print(f)
        
        file_path = os.path.join("LA/LA/ASVspoof2019_LA_train/flac", f)
        #print(file_path)
        file_path = file_path.replace("\\", "/")
        #add full path name
        ndarray, rate = lb.load(file_path, sr = None)
        melspectrogram = lb.feature.melspectrogram(y=ndarray, sr=rate, n_mels = mels)
        if melspectrogram.shape[1] > frames:
            melspectrogram = melspectrogram[:, :frames]
        else:
            melspectrogram = np.pad(melspectrogram, ((0, 0), (0, frames - melspectrogram.shape[1])), mode='constant')
        x_train.append(melspectrogram)
        y_train.append(label)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    return x_train, y_train




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
    #os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    training_dir_la = "LA/LA/ASVspoof2019_LA_train/flac"
    validation_dir_la = "LA/LA/ASVspoof2019_LA_dev/flac"
    testing_dir_la = "LA/LA/ASVspoof2019_LA_eval/flac"
    training_dir_pa = "PA/PA/ASVspoof2019_PA_train/flac"
    validation_dir_pa = "PA/PA/ASVspoof2019_PA_eval/flac"
    testing_dir_pa = "PA/PA/ASVspoof2019_PA_eval/flac"
    training_files_la = os.listdir(training_dir_la)
    validation_files_la = os.listdir(validation_dir_la)
    testing_files_la = os.listdir(testing_dir_la)
    training_files_pa = os.listdir(training_dir_pa)
    validation_files_pa = os.listdir(validation_dir_pa)
    testing_files_pa = os.listdir(testing_dir_pa)
    
    counter = 0
    counter1 = 0
    counter2 = 0
    counter3 = 0
    for x in training_files_la:
        counter += 1
    for x in validation_files_la:
        counter1 += 1
    # for x in training_files_pa:
    #     counter2 += 1
    # for x in testing_files_pa:
    #     counter3 += 1
    print(counter)
    print(counter1)
    # print(counter2)
    # print(counter3)

    x_train_training, y_train_training = create_melspectrogram(training_files_la, 128, 128)
    x_train_eval, y_train_eval = create_melspectrogram(validation_files_la, 128, 128)
    x_train_testing, y_train_testing = create_melspectrogram(testing_files_la, 128, 128)
    



if __name__ == "__main__":
    main()
