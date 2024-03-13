import librosa as lb
import numpy as np
import tensorflow as tf
import os
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#bonafide = 1, fake = 0
def create_melspectrogram(file_name, path, cm_path, mels, frames):
    x_train = []
    y_train = []
    for f in file_name:
        label = testing_real_or_fake(f, cm_path)
        #print(f)
        
        file_path = os.path.join(path, f)
        
        file_path = file_path.replace("\\", "/")
        print(file_path)
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




def testing_real_or_fake(file, cm_path):
    with open(cm_path, 'r') as f:
        for line in f:
            fields = line.split(" ")
            if fields[1] == file.rsplit(".", 1)[0]:
                if fields[4].rstrip("\n") == "bonafide":
                    return 1
                elif fields[4].rstrip("\n") == "spoof":
                    return 0
                
def create_cnn(input_shape):
    model = tf.keras.Sequential([tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')])

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    return model
def main():
    if os.path.exists('x_train_training.npy') and os.path.exists('y_train_training.npy') and os.path.exists('x_train_val.npy') and os.path.exists('y_train_val.npy') and os.path.exists('x_train_testing.npy') and os.path.exists('y_train_testing.npy'):
        x_train_training = np.load('x_train_training.npy', allow_pickle=True)
        y_train_training = np.load('y_train_training.npy', allow_pickle=True)
        x_train_val = np.load('x_train_val.npy', allow_pickle=True)
        y_train_val = np.load('y_train_val.npy', allow_pickle=True)
        x_train_testing = np.load('x_train_testing.npy', allow_pickle=True)
        y_train_testing = np.load('y_train_testing.npy', allow_pickle=True)
    else:
        training_dir_la = "LA/LA/ASVspoof2019_LA_train/flac"
        validation_dir_la = "LA/LA/ASVspoof2019_LA_dev/flac"
        testing_dir_la = "LA/LA/ASVspoof2019_LA_eval/flac"
        # training_dir_pa = "PA/PA/ASVspoof2019_PA_train/flac"
        # validation_dir_pa = "PA/PA/ASVspoof2019_PA_eval/flac"
        # testing_dir_pa = "PA/PA/ASVspoof2019_PA_eval/flac"
        training_files_la = os.listdir(training_dir_la)
        validation_files_la = os.listdir(validation_dir_la)
        testing_files_la = os.listdir(testing_dir_la)
        # training_files_pa = os.listdir(training_dir_pa)
        # validation_files_pa = os.listdir(validation_dir_pa)
        # testing_files_pa = os.listdir(testing_dir_pa)
        train_cm_path = "LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
        val_cm_path = "LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
        test_cm_path = "LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"

        x_train_training, y_train_training = create_melspectrogram(training_files_la, training_dir_la, train_cm_path, 128, 128)
        x_train_val, y_train_val = create_melspectrogram(validation_files_la, validation_dir_la, val_cm_path, 128, 128)
        x_train_testing, y_train_testing = create_melspectrogram(testing_files_la, testing_dir_la,test_cm_path,128, 128)
        np.save('x_train_training.npy', x_train_training)
        np.save('y_train_training.npy', y_train_training)
        np.save('x_train_val.npy', x_train_val)
        np.save('y_train_val.npy', y_train_val)
        np.save('x_train_testing.npy', x_train_testing)
        np.save('y_train_testing.npy', y_train_testing)
    
    print(f"x_train_training: {x_train_training}")
    print(f"y_train_testing: {y_train_training}")
    print(f"x_train_val: {x_train_val}")
    print(f"y_train_val: {y_train_val}")
    print(f"x_train_test: {x_train_testing}")
    print(f"y_train_test: {y_train_testing}")
    x_train_training = x_train_training.reshape(-1, 128, 128, 1)
    model = create_cnn(x_train_training.shape[1:])
    model.fit(x_train_training, y_train_training, epochs=10, batch_size=32, validation_data=(x_train_val, y_train_val))
    loss, accuracy = model.evaluate(x_train_testing, y_train_testing)
    print(accuracy)


if __name__ == "__main__":
    main()
