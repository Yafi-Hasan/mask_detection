import os
import cv2
import ssl
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.python.ops.gen_nn_ops import leaky_relu
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, Model
from keras.preprocessing.image import ImageDataGenerator
from cv2 import split, merge, createCLAHE


def cleaning(path):
    print('[INFO] Cleaning...')
    try:
        os.remove(os.path.join(path, '.DS_Store'))
    except FileNotFoundError or FileExistsError:
        pass
    print('[INFO] Cleaning completed')


def clahe(frame):
    r,g,b = split(frame)
    clahe = createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
    claher=clahe.apply(r)
    claheg=clahe.apply(g)
    claheb=clahe.apply(b)
    frame=merge((claher,claheg,claheb))
    return frame


def rename(source_path,target_path,file_number=1):
    print('[INFO] Renaming files...')
    n = file_number
    try:
        os.listdir(source_path)
    except FileNotFoundError :
        print('[ALERT!] Source path not found')

    try:
        os.makedirs(target_path)
    except FileExistsError:
        pass

    cleaning(source_path)
    for count, filename in enumerate(os.listdir(source_path)):
        dst = str(n) + '.jpg'
        os.rename(os.path.join(source_path,filename), os.path.join(target_path,dst))
        n = n+1
    print('[INFO] File sucessfully renamed')


def main_process(source_path, target_path, image_size):
    print('[INFO] Processing files...')
    try:
        os.listdir(source_path)
    except FileNotFoundError :
        print('[ALERT!] Source path not found')
        exit()

    try:
        os.makedirs(target_path)
    except FileExistsError :
        pass

    cleaning(source_path)
    for count, filename in enumerate(os.listdir(source_path)):
        image = cv2.imread(os.path.join(source_path,filename))
        imres = cv2.resize(image,(image_size,image_size))
        imres = clahe(imres)
        cv2.imwrite(os.path.join(target_path, filename),imres)
    print('[INFO] File successfully resize, and CLAHE histogram equalization')


def prep_data(train_path,test_path,validation_path):
    print('[INFO] Running data preparation...')
    print('[INFO] Searching for path...')
    try:
        os.listdir(train_path)
        data_dir = train_path
        print('[INFO] Train data path found')
        os.listdir(test_path)
        test_dir = test_path
        print('[INFO] Test data path found')
        os.listdir(validation_path)
        print('[INFO] Validation path found')
    except FileNotFoundError :
        print('[ALERT!] Folder not found')
    print('[INFO] Creating train data...')
    traingen = ImageDataGenerator(rotation_range=25,zoom_range=0.1,shear_range=0.15,width_shift_range=0.15,height_shift_range=0.15,horizontal_flip=True)
    train_data = traingen.flow_from_directory(data_dir,target_size=(224,224),color_mode='rgb',class_mode='categorical',shuffle=True,batch_size=512)
    print('[INFO] Train data created')
    print('[INFO] Creating test data...')
    testgen = ImageDataGenerator()
    test_data = testgen.flow_from_directory(test_dir,target_size=(224,224),color_mode='rgb',class_mode='categorical',shuffle=True,batch_size=512)
    print('[INFO] Test data created')
    print('[INFO] Creating validating data...')
    valgen = ImageDataGenerator()
    val_data = valgen.flow_from_directory(validation_path,target_size=(224,224),color_mode='rgb',class_mode='categorical',shuffle=True)
    print('[INFO] Validation data created')
    print('[INFO] Data preparation completed')
    return train_data, test_data, val_data


def custom_mobilenet_v2(training_set, testing_set, validation_set):
    print('[INFO] Running custom model creation...')
    ssl._create_default_https_context = ssl._create_unverified_context
    print('[INFO] Accessing architecture...')
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3), classes=3)
    print('[INFO] Modifying architecture...')
    base_model.trainable = False
    pool_layer = layers.GlobalAveragePooling2D()(base_model.output)
    flat_layer = layers.Flatten()(pool_layer)
    custom_1 = layers.Dense(1024)(flat_layer)
    custom_1 = layers.Activation(leaky_relu)(custom_1)
    custom_2 = layers.Dense(500)(custom_1)
    custom_2 = layers.Activation(leaky_relu)(custom_2)
    custom_3 = layers.Dense(1024)(custom_2)
    custom_3 = layers.Activation(leaky_relu)(custom_3)
    final_layer = layers.Dense(3)(custom_3)
    final_layer = layers.Activation('softmax')(final_layer)
    new_model = Model(inputs=base_model.input, outputs=final_layer)
    new_model.summary()

    print('[INFO] Custom model created')
    print('[INFO] Compiling model...')
    new_model.compile(loss='categorical_crossentropy', metrics=['Accuracy'],
                      optimizer=Adam(learning_rate=0.05))
    print('[INFO] Training model...')
    es = EarlyStopping(monitor='val_Accuracy', patience=25, min_delta=0.005, mode='max', verbose=1,
                       restore_best_weights=True)
    history = new_model.fit(training_set, epochs=30, validation_data=testing_set, callbacks=[es])
    new_model.evaluate(validation_set, verbose=1, callbacks=[es])
    target_file = os.path.join(os.getcwd(), 'NN Model')
    try:
        print('[INFO] Creating target file...')
        os.makedirs(target_file)
        print(f'[INFO] Target path successfully build')
    except FileExistsError:
        print(f'[INFO] Target path already exist')

    plt.plot(history.history['Accuracy'])
    plt.plot(history.history['val_Accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    print('[INFO] Saving model...')
    new_model.save(os.path.join(target_file, 'face_model.model'))
    new_model.save_weights(os.path.join(target_file, 'Weights'))
    print('[INFO] Model Saved')
    print('[INFO] Custom MobileNet-V2 model successfully saved')
    return new_model


def convert_model(modelpath, target_path):
    print('[INFO] Converting model to lite mode...')
    try:
        os.listdir(modelpath)
        set = True
    except FileNotFoundError:
        print('[ALERT!] Model path not found')
        set = False

    if set:
        converting = tf.lite.TFLiteConverter.from_saved_model(modelpath)
        tflite_model = converting.convert()
        try:
            os.makedirs(target_path)
        except FileExistsError:
            pass
        open(os.path.join(target_path, 'Model.tflite'), 'wb').write(tflite_model)
        print('[INFO] TF lite model successfully converted and saved.')


def main_program():
    # Processing files
    work_dir = os.getcwd()
    dataset_dir = os.path.join(work_dir,'Raw Dataset')
    kelas = ['Masked', 'Unmasked', 'Unproper']
    for set in kelas:
        cleaning(os.path.join(dataset_dir,set))
        target_dir = os.path.join(os.path.join(work_dir, 'Raw Edited'), set)
        main_process(os.path.join(dataset_dir,set),target_dir,224)
        fintar_dir = os.path.join(os.path.join(work_dir, 'Dataset'), set)
        rename(target_dir,fintar_dir,file_number=1)
    shutil.rmtree(os.path.join(work_dir, 'Raw Edited'))
    new_data_dir = os.path.join(work_dir, 'Dataset')
    for i in range(1,664):
        for set in kelas:
            source_dir = os.path.join(new_data_dir, set)
            newtar_dir = os.path.join(os.path.join(new_data_dir, 'Training'), set)
            try:
                os.makedirs(newtar_dir)
            except FileExistsError:
                pass
            shutil.move(os.path.join(source_dir, str(i)+'.jpg'), newtar_dir)
    for i in range(664,742):
        for set in kelas:
            source_dir = os.path.join(new_data_dir, set)
            test_tar_dir = os.path.join(os.path.join(new_data_dir, 'Testing'), set)
            try:
                os.makedirs(test_tar_dir)
            except FileExistsError:
                pass
            shutil.move(os.path.join(source_dir, str(i)+'.jpg'), test_tar_dir)
    for set in kelas:
        sor_dir = os.path.join(new_data_dir, set)
        val_dir = os.path.join(new_data_dir, 'Validation')
        try:
            os.makedirs(val_dir)
        except FileExistsError:
            pass
        shutil.move(sor_dir, val_dir)

    # Creating model
    train_path = os.path.join(os.path.join(work_dir,'Dataset'),'Training')
    test_path = os.path.join(os.path.join(work_dir,'Dataset'), 'Testing')
    val_path = os.path.join(os.path.join(work_dir, 'Dataset'), 'Validation')
    train_set, test_set, val_set = prep_data(train_path, test_path, val_path)
    model = custom_mobilenet_v2(train_set, test_set, val_set)

    # Converting model to lite mode
    model_dir = os.path.join(os.path.join(work_dir, 'NN Model'), 'face_model.model')
    tar_lite_dir = os.path.join(work_dir, 'Lite Model')
    convert_model(model_dir, tar_lite_dir)


if __name__ == '__main__':
    main_program()