# Semillas
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

# Imports básicos
import numpy as np
import keras
import matplotlib.pyplot as plt

# Cargar capas y modelos
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Flatten, BatchNormalization

# Útiles
import keras.utils as np_utils
# Optimizador
from keras.optimizers import SGD
# Conjunto datos
from keras.datasets import cifar100

# Preprocesador de imágenes
from keras.preprocessing.image import ImageDataGenerator

# Pesos por si se reutiliza un modelo
weights = None

#########################################################################
######## FUNCIÓN PARA CARGAR Y MODIFICAR EL CONJUNTO DE DATOS ###########
#########################################################################

"""
    Devuelve 4 vectores conteniendo por este orden: las imágenes de entrenamiento,
    las clases de las imágenes de entrenamiento, las imágenes del conjunto de test
    y las clases del conjunto de test.
"""
def cargarImagenes():
    # Tamaño (32, 32, 3), imagenes de 25 de las clases
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255

    # Tomamos 25 clases de todas las que hay
    train_idx = np.isin(y_train, np.arange(25))
    train_idx = np.reshape(train_idx, -1)
    x_train = x_train[train_idx]
    y_train = y_train[train_idx]

    test_idx = np.isin(y_test, np.arange(25))
    test_idx = np.reshape(test_idx, -1)
    x_test = x_test[test_idx]
    y_test = y_test[test_idx]

    # Transformamos los vectores de clases en matrices.
    y_train = np_utils.to_categorical(y_train, 25)
    y_test = np_utils.to_categorical(y_test, 25)

    return x_train, y_train, x_test, y_test

#########################################################################
######## FUNCIÓN PARA OBTENER EL ACCURACY DEL CONJUNTO DE TEST ##########
#########################################################################

"""
    Devuelve la accuracy de un modelo, que es el % de etiquetas bien predichas
    frente al total. Se le pasa el vector de etiquetas real y el predicho.
"""
def calcularAccuracy(labels, preds):
    labels = np.argmax(labels, axis = 1)
    preds = np.argmax(preds, axis = 1)

    accuracy = sum(labels == preds) / len(labels)

    return accuracy

#########################################################################
## FUNCIÓN PARA PINTAR LA PÉRDIDA Y EL ACCURACY EN TRAIN Y VALIDACIÓN ###
#########################################################################

"""
    Pinta dos gráficas: la evolución de la función de perdida y la evolución
    del accuracy tanto en train como en validación. Hay que pasarle el historial
    de entrenamiento (funciones fit() y fir_generator())
"""
def mostrarEvolucion(hist):
    # Evolucion función de perdida
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['Training loss', 'Validation loss'])
    plt.show()

    # Evolución de accuracy
    acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(['Training accuracy', 'Validation accuracy'])
    plt.show()

#########################################################################
################## DEFINICIÓN DEL MODELO BASENET ########################
#########################################################################
"""
    Definición del modelo de BaseNet Básico (Ej1)
"""
def baseNetBasico():
    # Modelo sequencial
    my_model = Sequential()
    # Conv2D | Ksize = 5 | I/O dim = 32,28 | I/O channels = 3,6 | Act = ReLu
    my_model.add(Conv2D(6, 5, input_shape = (32, 32, 3), activation = "relu"))
    # MaxPooling2D | Ksize = 2 | I/O dim = 28,14
    my_model.add(MaxPooling2D((2,2)))
    # Conv2D | Ksize = 5 | I/O dim = 14,10 | I/O channels = 6,16 | Act = ReLu
    my_model.add(Conv2D(16, 5, activation = "relu"))
    # MaxPooling2D | Ksize = 2 | I/O dim = 10, 5
    my_model.add(MaxPooling2D((2,2)))
    # Flatten
    my_model.add(Flatten())
    # Linear | I/O dim = 400, 50 | Act = Relu
    my_model.add(Dense(50, activation = "relu"))
    # Linear 50 | 25 | Act = SoftMax
    my_model.add(Dense(25, activation = "softmax"))
    return my_model

"""
    Definición del modelo de BaseNet con BatchNormalization (Ej2)
"""
def baseNetNormalizado():
    # Modelo sequencial
    my_model = Sequential()
    # Conv2D | Ksize = 5 | I/O dim = 32,28 | I/O channels = 3,6 | Act = ReLu
    my_model.add(Conv2D(6, 5, input_shape = (32, 32, 3)))
    my_model.add(Activation("relu"))
    # BatchNormalization
    my_model.add(BatchNormalization())
    # MaxPooling2D | Ksize = 2 | I/O dim = 28,14
    my_model.add(MaxPooling2D((2,2)))
    # Conv2D | Ksize = 5 | I/O dim = 14,10 | I/O channels = 6,16 | Act = ReLu
    my_model.add(Conv2D(16, 5))
    my_model.add(Activation("relu"))
    # BatchNormalization
    my_model.add(BatchNormalization())
    # MaxPooling2D | Ksize = 2 | I/O dim = 10, 5
    my_model.add(MaxPooling2D((2,2)))
    # Flatten
    my_model.add(Flatten())
    # Linear | I/O dim = 400, 50 | Act = Relu
    my_model.add(Dense(50))
    my_model.add(Activation("relu"))
    # BatchNormalization
    my_model.add(BatchNormalization())
    # Linear 50 | 25 | Act = SoftMax
    my_model.add(Dense(25))
    my_model.add(Activation("softmax"))
    return my_model

#########################################################################
######### DEFINICIÓN DEL OPTIMIZADOR Y COMPILACIÓN DEL MODELO ###########
#########################################################################
"""
    Compilamos el modelo "my_model" con el optimizador "SGD" y la función
    de pérdida "categorical_crossentropy" para clasificar varias clases.
"""
def compilarModelo(my_model):
    # Optimizador
    opt = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
    # Compilamos el modelo
    my_model.compile(opt, loss = "categorical_crossentropy", metrics = ["acc"])
    return my_model

#########################################################################
###################### ENTRENAMIENTO DEL MODELO #########################
#########################################################################
"""
    Entrena el modelo "my_model" usando el preprocesador de imágenes "datagen"
    Si "datagen" es None, se aplica un entrenamiento normal (ej1).
    Se pueden cambiar "batch_size", "epochs" y "verbose"; por defecto
    vienen respectivamente con 32, 30 y 1.
    Devuelve la historia de entrenamiento e imprime el resultado de la
    precisión y el error.
"""
def trainModel(my_model, datagen = None, batch_size = 32, epochs = 15, verbose=1):
    # Compilar el modelo
    my_model = compilarModelo(my_model)
    # Salvamos los pesos por si reutilizamos el modelo
    weights = my_model.get_weights()
    # Cargamos el dataset
    x_train, y_train, x_test, y_test = cargarImagenes()
    # Si tenemos un preprocesador
    if datagen != None:
      # Aplicamos el preprocesamiento
      datagen.fit(x_train)
      datagen.standardize(x_test)
      # Generador de entrenamiento
      train_generator = datagen.flow(x_train,
                                     y_train,
                                     batch_size = batch_size,
                                     subset = "training")
      # Generador de validación
      validation_generator = datagen.flow(x_train,
                                          y_train,
                                          batch_size = batch_size,
                                          subset = "validation")
      # Entrenamos el modelo
      history = my_model.fit_generator(train_generator,
                                       steps_per_epoch = len(x_train) * 0.9 / batch_size,
                                       epochs = epochs,
                                       validation_data = validation_generator,
                                       validation_steps = len(x_train) * 0.1 / batch_size,
                                       verbose = verbose)
      # Evaluamos el modelo
      prediccions = my_model.predict_generator(datagen.flow(x_test,
                                                            batch_size = 1,
                                                            shuffle = False),
                                               steps = len(x_test))
    else:
      # Entrenamos el modelo
      history = my_model.fit(x_train, y_train,
                            validation_split = 0.1,
                            batch_size = batch_size,
                            epochs = epochs,
                            verbose = verbose)
      # Evaluamos el modelo
      prediccions = my_model.predict(x_test, batch_size = batch_size, verbose = verbose)
    # Calculamos la accuracy
    acc = calcularAccuracy(y_test, prediccions)
    # Imprimimos la precisión en conjunto test
    print("Acc: " + str(acc))
    # Restauramos los pesos por si usamos el mismo modelo
    my_model.set_weights(weights)
    # Devolvemos la historia de entrenamiento
    return history

#########################################################################
########################### PREPROCESADORES #############################
#########################################################################

"""
    Preprocesador básico, no modifica nada, solo divide un 10% para
    el conjunto de validación.
"""
def preProcesadorBase():
    return ImageDataGenerator(validation_split = 0.1)

"""
    Preprocesador para normalizar los datos (media 0 y desviación 1)
    y también divide en 10% para validación.
"""
def preProcesadorNormalizado():
    return ImageDataGenerator(featurewise_center = True,
                              featurewise_std_normalization = True,
                              validation_split = 0.1)

"""
    Preprocesador para "aumentar" las imágenes, crea más imagenes a
    partir de las que ya hay. Los parámetros que se le pasan son los
    siguientes "rotation_range", "width_shift_range", "height_shift_range",
    "zoom_range", "horizontal_flip", "vertical_flip" (por defecto no hacen
    nada). Además divide en un 10% para validación
"""
def preProcesadorAugmentedData(rotation_range = 0, width_shift_range = 0,
                                height_shift_range = 0, zoom_range = 0,
                                horizontal_flip = False, vertical_flip = False):
    return ImageDataGenerator(rotation_range = rotation_range,
                              width_shift_range = width_shift_range,
                              height_shift_range = height_shift_range,
                              zoom_range = zoom_range,
                              horizontal_flip = horizontal_flip,
                              vertical_flip = vertical_flip,
                              validation_split = 0.1)

"""
    Modelo básico de RESNET sin preprocesamiento. Parámetros de entrenamiento por defecto.
"""
def ej1():
  # Modelo básico
  my_model = baseNetBasico()
  # Entrenamos
  history = trainModel(my_model)
  # Mostramos evolución
  mostrarEvolucion(history)
  # Validación con ImageDataGenerator
  history = trainModel(my_model, preprocesadorBase())
  mostrarEvolucion(history)

def ej2():
    print("BaseNET con normalización")
    # Creamos el modelo
    my_model = baseNetBasico()
    # LO mostramos
    my_model.summary()
    # ImageData para normalizar
    datagen = preprocesadorNormalizado()
    # Entrenamos y obtenemos el historial de entrenamiento
    history = trainModel(my_model, datagen)
    # Mostramos la evolución
    mostrarEvolucion(history)

    # BaseNET básico

if __name__ == "__main__":
    ej1()
    #ej2()
