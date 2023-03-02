import tensorflow as tf
import seaborn as sns
from keras import layers
from keras.models import Sequential

from tkinter import *
import numpy as np
import matplotlib.pyplot as plt


def main():

    epocas = int(cantidad_epocas.get())

    batch_size = 180 # cantidad de imagenes
    img_height = 100 # alto
    img_width = 100 # ancho

    train_ds = tf.keras.utils.image_dataset_from_directory("./data/Entrenamiento",image_size=(img_height, img_width),batch_size=batch_size)
    val_ds = tf.keras.utils.image_dataset_from_directory("./data/Validacion",image_size=(img_height, img_width),batch_size=batch_size)
    class_names = train_ds.class_names
    print(class_names)

    train_ds = train_ds.cache().prefetch(buffer_size=10)
    val_ds = val_ds.cache().prefetch(buffer_size=10)

    num_clases = len(class_names)
    data_augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal",input_shape=(img_height,img_width,3)),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),])

    modelo = Sequential([
        data_augmentation,
            layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(3,3),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(3,3),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(3,3),
            # layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(50, activation='relu'),
            # layers.Dense(50, activation='relu'),
            layers.Dense(num_clases)
        ])

    modelo.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    modelo.summary()

    #epochs = 50
    history = modelo.fit(train_ds,validation_data=val_ds,epochs=epocas)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(1,epocas+1)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Precision de entrenamiento')
    plt.plot(epochs_range, val_acc, label='Precision de validación')
    plt.legend(loc='lower right')
    plt.title('Presicion de entrenamiento y validacion')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Error de entrenamiento')
    plt.plot(epochs_range, val_loss, label='Error de validacion')
    plt.legend(loc='upper right')
    plt.title('Error de entrenamiento y validacion')
    plt.show()

    for i in range(5):
        for images, labels in val_ds.take(1):
            predictions = modelo.predict(images)
            score = tf.nn.softmax(predictions[i])
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title('Clase: {} | Precisión: {:.2f}'.format(class_names[np.argmax(score)], 100 * np.max(score)))
            plt.show()
            break

    test_labels = []
    test_images = []
    for img, labels in val_ds.take(1):
        test_images.append(img)
        test_labels.append(labels)

    y_pred = np.argmax(modelo.predict(test_images), axis=1).flatten()
    y_true = np.asarray(test_labels).flatten()
    test_acc = sum(y_pred == y_true) / len(y_true)
    print(("Test accuracy: {:.2f}%".format(test_acc * 100)))


    consfusion_matrix = tf.math.confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 10))
    sns.heatmap(consfusion_matrix.numpy(), 
        xticklabels=class_names,
        yticklabels=class_names, 
        annot=True, fmt="d")
    
    plt.title('Matriz de confusion')
    plt.xlabel('Predicciones')
    plt.ylabel('Datos')
    plt.show()


ventana = Tk()
ventana.title('IA - Actividad 3, Corte 2')
ventana.geometry('400x200')

label = Label(ventana, text='Ingrese cantidad de epocas de entrenamiento:').place(x=80, y=10)
cantidad_epocas = Entry(ventana)
cantidad_epocas.place(x=140, y=60)
button = Button(ventana, text='Aceptar', command=lambda:main()).place(x=180, y=100)

ventana.mainloop()