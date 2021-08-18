import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import seaborn as sn
import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix

import pathlib

batch_size = 32
img_height = 180
img_width = 180

if False:
    data_dir= pathlib.Path("./static/database")

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)


    class_names = train_ds.class_names
    print(class_names)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


    normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixels values are now in `[0,1]`.

    num_classes = 3

    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal",
                                                        input_shape=(img_height,
                                                                    img_width,
                                                                    3)),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
            layers.experimental.preprocessing.RandomContrast(0.2)
        ]
    )

    model = Sequential([
        data_augmentation,
        layers.experimental.preprocessing.Rescaling(1./255),
        layers.AveragePooling2D(6,3),
        #layers.Conv2D(16, 3, padding='same', activation='relu'),
        #layers.MaxPooling2D(),
        layers.Conv2D(32, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128,(3,3), padding='same',activation='relu'),
        layers.MaxPooling2D(),
        #layers.Conv2D(256, (5,5), padding='same', activation='relu'),
        #layers.MaxPooling2D(),
        #layers.Conv2D(512,(5,5), padding='same',activation='relu'),
        #layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes,activation='softmax',kernel_regularizer='l1')
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])



    epochs = 110
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        batch_size=32
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


    model.save('saved_model/test_model_200_128x128_1.h5')

################################################
################## TEST DATA ###################
################################################
if False :
    test_data_dir=pathlib.Path("./static/test_database")
    model = tf.keras.models.load_model('./saved_model/test_model_200_128x128_1.h5')
    class_names = ['Melanoma','Nevus','Seborrheic Keratosis']
    test_labels=[]
    files=[]
    for r, d, f in os.walk("./static/tester/"):
        for file in f:
            if ('.jpg' in file):
                exact_path = r + file
                if (exact_path.find("melanoma")!= -1):
                    test_labels.append("Melanoma")
                if (exact_path.find("nevus")!= -1):
                    test_labels.append("Nevus")
                if (exact_path.find("seborrheic")!= -1):
                    test_labels.append("Seborrheic Keratosis")
                files.append(exact_path)
    test_preds=[]
    for img_path in files:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(128, 128))
        img_arr = tf.keras.preprocessing.image.img_to_array(img)
        img_arr = np.array([img_arr])
        predictions = model.predict(img_arr)
        score = tf.nn.softmax(predictions[0])    
        prediction = str(class_names[np.argmax(score)])
        test_preds.append(prediction)

    cm=confusion_matrix(test_labels,test_preds,labels=["Melanoma", "Nevus","Seborrheic Keratosis"])
    #print(cm)
    #plt.imshow(cm, cmap='binary')
    df_cm = pd.DataFrame(cm, index = class_names,
                    columns = class_names)
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.show()

#print(test_ds[0].class_names)

#test_preds=model.predict(test_ds)
#plot_confusion(y_true=test_ds, y_pred=test_preds, labels=test_ds.class_names, figsize=(6,4))


#for img_path in files:
#    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(128, 128))
#    img_arr = tf.keras.preprocessing.image.img_to_array(img)
#    img_arr = np.array([img_arr]) 

#AUTOTUNE = tf.data.AUTOTUNE
#test_ds= test_ds.cache().prefetch(buffer_size=AUTOTUNE)

