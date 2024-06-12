import librosa as lb
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import deepfake_audio  

def create_cnn(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape = input_shape), 
        tf.keras.layers.Conv2D(64, kernel_size = (5,5), activation="selu"),
        tf.keras.layers.MaxPooling2D(pool_size = (2,2)),
        tf.keras.layers.Conv2D(128, kernel_size = (3,3), activation="selu"),
        tf.keras.layers.MaxPooling2D(pool_size = (2,2)),
        tf.keras.layers.Conv2D(256, kernel_size = (3,3), activation="selu"),
        tf.keras.layers.MaxPooling2D(pool_size = (2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="selu"),
        tf.keras.layers.Dropout(.5), 
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer="adam",
                loss='binary_crossentropy',
                metrics=['accuracy'])
    
    return model

def main():
    # model = tf.keras.models.load_model("deepfake_audio_model.h5")
    # model.compile(optimizer=tf.keras.optimizers.Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    training_dir_pa = "PA/PA/ASVspoof2019_PA_train/flac"
    validation_dir_pa = "PA/PA/ASVspoof2019_PA_dev/flac"
    testing_dir_pa = "PA/PA/ASVspoof2019_PA_eval/flac"
    training_files_pa = os.listdir(training_dir_pa)
    validation_files_pa = os.listdir(validation_dir_pa)
    testing_files_pa = os.listdir(testing_dir_pa)
    train_cm_path_pa = "PA/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.train.trn.txt"
    val_cm_path_pa = "PA/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.dev.trl.txt"
    test_cm_path_pa = "PA/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.eval.trl.txt"

    if os.path.exists('x_train_testing_pa.npy') and os.path.exists('y_train_testing_pa.npy'):
        x_train_testing_pa = np.load('x_train_testing_pa.npy', allow_pickle=True)
        y_train_testing_pa = np.load('y_train_testing_pa.npy', allow_pickle=True)
    else:
        x_train_testing_pa, y_train_testing_pa = deepfake_audio.create_melspectrogram(testing_files_pa, testing_dir_pa, test_cm_path_pa, 128, 128)
        np.save('x_train_testing_pa.npy', x_train_testing_pa)
        np.save('y_train_testing_pa.npy', y_train_testing_pa)

    if os.path.exists('x_train_training_pa.npy') and os.path.exists('y_train_training_pa.npy'):
        x_train_training_pa = np.load('x_train_training_pa.npy', allow_pickle=True)
        y_train_training_pa = np.load('y_train_training_pa.npy', allow_pickle=True)
    else:
        x_train_training_pa, y_train_training_pa = deepfake_audio.create_melspectrogram(training_files_pa, training_dir_pa, train_cm_path_pa, 128, 128)
        np.save('x_train_training_pa.npy', x_train_training_pa)
        np.save('y_train_training_pa.npy', y_train_training_pa)

    if os.path.exists('x_train_val_pa.npy') and os.path.exists('y_train_val_pa.npy'):
        x_train_val_pa = np.load('x_train_val_pa.npy', allow_pickle=True)
        y_train_val_pa = np.load('y_train_val_pa.npy', allow_pickle=True)
    else:
        x_train_val_pa, y_train_val_pa = deepfake_audio.create_melspectrogram(validation_files_pa, validation_dir_pa, val_cm_path_pa, 128, 128)
        np.save('x_train_val_pa.npy', x_train_val_pa)
        np.save('y_train_val_pa.npy', y_train_val_pa)

    x_train_training_pa = x_train_training_pa.reshape(-1, 128, 128, 1)
    model1 = create_cnn(x_train_training_pa.shape[1:])
    model1.fit(x_train_training_pa, y_train_training_pa, epochs=1, batch_size = 128, validation_data=(x_train_val_pa, y_train_val_pa))
    loss_pa, accuracy_pa = model1.evaluate(x_train_testing_pa, y_train_testing_pa)
    model1.save("deepfake_audio_model_pa.h5")
    # model.fit(x_train_training_pa, y_train_training_pa, epochs=1, batch_size= 64, validation_data=(x_train_val_pa, y_train_val_pa))
    # loss_pa, accuracy_pa = model.evaluate(x_train_testing_pa, y_train_testing_pa)

    print(f"Physical Access Accuracy: {accuracy_pa}")
    print(f"Physical Access Loss: {loss_pa}")

if __name__ == "__main__":
    main()