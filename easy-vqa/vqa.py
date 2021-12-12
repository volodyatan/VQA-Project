## for model building
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.backend import argmax
from tensorflow import keras
import tensorflow as tf

## imports from our files
from data_prep import get_data

## helpers
import pickle
import os.path

def init_model(img_shape, vocab_size, num_answers):

    ## convolutional NN for images
    img_input = Input(shape = img_shape, name ='image_input')
    
    img_info = Conv2D(8, 3, padding='same')(img_input)
    img_info = MaxPooling2D(padding='same')(img_info)

    img_info = Conv2D(16, 3, padding='same')(img_info)
    img_info = MaxPooling2D(padding='same')(img_info)

    img_info = Conv2D(32, 3, padding='same')(img_info)
    img_info = MaxPooling2D(padding='same')(img_info)

    img_info = Flatten()(img_info)
    
    img_info = Dense(32, activation ='swish')(img_info)

    q_input = Input(shape=(vocab_size,), name = 'question_input')
    q_info = Dense(32, activation='swish')(q_input)
    q_info = Dense(32, activation='swish')(q_info)


    ## merge img_info and q_info
    output = Multiply()([img_info, q_info])
    output = Dense(32, activation='swish')(output)
    output = Dense(num_answers, activation='softmax')(output)

    model = Model(inputs=[img_input, q_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=5e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model


def run_model(epochs, model_name):
    if os.path.isfile('./model_data/{}_data'.format(model_name)):
        with open('./model_data/{}_data'.format(model_name), 'rb') as data_file:
            data = pickle.load(data_file)
        
    else:
        data = get_data()
        with open('./model_data/{}_data'.format(model_name), 'wb') as data_file:
            pickle.dump(data, data_file)
    data_file.close()
    train_imgs, test_imgs, train_answers, test_answers, train_qs, test_qs, possible_answers, num_words, img_shape = data

    num_answers = len(possible_answers)

    # if model exists, retrieve it
    if os.path.isdir('./models/'+model_name):
        print('LOADING MODEL.......')
        model = load_model('./models/'+model_name)
    else:
        model = init_model(img_shape, num_words, num_answers)

    # save model every epoch
    checkpoint = keras.callbacks.ModelCheckpoint('./models/'+model_name, period=5) 

    print('fitting...')
    history = model.fit(x = [train_imgs, train_qs],
        y=train_answers,
        batch_size=32,
        epochs=epochs,
        verbose=1,
        validation_data=([test_imgs,test_qs],test_answers),
        validation_steps=10,
        callbacks=[checkpoint]
    )

    with open('./history/{}_hist'.format(model_name), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    file_pi.close()

    model.save('./models/'+model_name)


run_model(epochs=20, model_name='easy-vqa')
