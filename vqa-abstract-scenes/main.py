from tensorflow import keras

## for model building
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Multiply, LSTM, Dropout
from tensorflow.keras.optimizers import Adam, SGD
## allows me to choose devices for training
from tensorflow import device

PREP = 'seq'
## imports from our files
from bow_data_prep import bow_get_data
from seq_data_prep import seq_get_data

## for dealing with images
from skimage.io import imread
from skimage.transform import resize
import os.path

## helpers
import numpy as np
import pickle

IMG_SHAPE = (200, 350, 3)
BATCH_SIZE = 128
EPOCHS = 10

# construct the training image generator for data augmentation
class Standard_Generator(keras.utils.Sequence) :
  
  def __init__(self, image_filenames, questions, labels, batch_size) :
    self.image_filenames = image_filenames
    self.questions = questions
    self.labels = labels
    self.batch_size = batch_size
    
    
  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx) :
    batch_imgs = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]

    batch_qs = self.questions[idx * self.batch_size : (idx+1) * self.batch_size]

    batch_labels = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
    
    return [np.array([
            resize(imread(file_name), IMG_SHAPE) ## image resize is hardcoded
               for file_name in batch_imgs])/255.0, batch_qs], np.array(batch_labels)


# construct the training image generator for data augmentation
class Sequence_Generator(keras.utils.Sequence) :
  
  def __init__(self, image_filenames, questions, labels, batch_size) :
    self.image_filenames = image_filenames
    self.questions = questions
    self.labels = labels
    self.batch_size = batch_size
    
    
  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx) :
    batch_imgs = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]

    batch_qs = self.questions[idx * self.batch_size : (idx+1) * self.batch_size]

    batch_qs = np.reshape(batch_qs,(batch_qs.shape[0], 1, batch_qs.shape[1]))

    batch_labels = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
    
    return [np.array([
            resize(imread(file_name), IMG_SHAPE) ## image resize is hardcoded
               for file_name in batch_imgs])/255.0, batch_qs], np.array(batch_labels)



def init_sequential_model(img_shape, max_len_q, num_answers, layers):

    ## convolutional NN for images
    img_input = Input(shape = img_shape, name ='image_input')
    
    if layers >= 1:
      img_info = Conv2D(8, 3, padding='same')(img_input) ## may need to pad conv2D
      img_info = MaxPooling2D(padding='same')(img_info)

    if layers >= 2:
      img_info = Conv2D(16, 3, padding='same')(img_info)
      img_info = MaxPooling2D(padding='same')(img_info)

    if layers >= 3:
      img_info = Conv2D(32, 3, padding='same')(img_info)
      img_info = MaxPooling2D(padding='same')(img_info)

    if layers >= 4:
      img_info = Conv2D(64, 3, padding='same')(img_info)
      img_info = MaxPooling2D(padding='same')(img_info)

    if layers >= 5:
      img_info = Conv2D(128, 3, padding='same')(img_info)
      img_info = MaxPooling2D(padding='same')(img_info)

    img_info = Flatten()(img_info)
    
    img_info = Dense(64, activation ='swish', kernel_initializer='he_uniform')(img_info)

    q_input = Input(shape=(1, max_len_q), name = 'question_input')
    q_info = LSTM(64, dropout=0.2, input_shape=(1, max_len_q), name = "lstm_layer")(q_input)

    ## merge img_info and q_info
    output = Multiply()([img_info, q_info])
    output = Dense(64, activation='swish')(output)
    output = Dense(num_answers, activation='softmax')(q_info)

    model = Model(inputs=[img_input, q_input], outputs=output)
    model.compile(optimizer=SGD(
    learning_rate=5e-4, momentum=0.9, nesterov=False, name="SGD"
    ),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model

def init_model(img_shape, vocab_size, num_answers, layers):

    ## convolutional NN for images
    img_input = Input(shape = img_shape, name ='image_input')
    
    if layers >= 1:
      img_info = Conv2D(8, 3, padding='same')(img_input) ## may need to pad conv2D
      img_info = MaxPooling2D(padding='same')(img_info)

    if layers >= 2:
      img_info = Conv2D(16, 3, padding='same')(img_info)
      img_info = MaxPooling2D(padding='same')(img_info)

    if layers >= 3:
      img_info = Conv2D(32, 3, padding='same')(img_info)
      img_info = MaxPooling2D(padding='same')(img_info)

    if layers >= 4:
      img_info = Conv2D(64, 3, padding='same')(img_info)
      img_info = MaxPooling2D(padding='same')(img_info)

    if layers >= 5:
      img_info = Conv2D(128, 3, padding='same')(img_info)
      img_info = MaxPooling2D(padding='same')(img_info)

    img_info = Flatten()(img_info)
    
    img_info = Dense(32, activation ='swish', kernel_initializer='he_uniform')(img_info)

    q_input = Input(shape=(vocab_size,), name = 'question_input')
    q_info = Dense(32, activation='swish')(q_input)
    q_info = Dense(32, activation='swish')(q_info)

    ## merge img_info and q_info
    output = Multiply()([img_info, q_info])
    output = Dense(32, activation='swish')(output)
    q_info = Dropout(0.2)
    output = Dense(num_answers, activation='softmax')(output)
    
    model = Model(inputs=[img_input, q_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=5e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model

def run_model(epochs, model_name, layers, check_top_ans, removeYesNo, get_data, type):
  if os.path.isfile('./model_data/{}_data'.format(model_name)):
    with open('./model_data/{}_data'.format(model_name), 'rb') as data_file:
      data = pickle.load(data_file)
    
  else:
    data = get_data(check_top_ans, removeYesNo)
    with open('./model_data/{}_data'.format(model_name), 'wb') as data_file:
      pickle.dump(data, data_file)
  data_file.close()
  train_imgs, test_imgs, train_answers, test_answers, train_qs, test_qs, possible_answers, num_words = data

  num_answers = len(possible_answers)

  if type == "bow":
    my_training_batch_generator = Standard_Generator(train_imgs, train_qs, train_answers, batch_size=BATCH_SIZE)
    my_validation_batch_generator = Standard_Generator(test_imgs, test_qs, test_answers, batch_size=BATCH_SIZE)
  else:
    my_training_batch_generator = Sequence_Generator(train_imgs, train_qs, train_answers, batch_size=BATCH_SIZE)
    my_validation_batch_generator = Sequence_Generator(test_imgs, test_qs, test_answers, batch_size=BATCH_SIZE)

  # if model exists, retrieve it
  if os.path.isdir('./models/'+model_name):
    print('LOADING MODEL.......')
    model = load_model('./models/'+model_name)
  else:
    # initialize models for parameters
    if type == 'bow':
      model = init_model(IMG_SHAPE, num_words, num_answers, layers)
    else:
      model = init_sequential_model(IMG_SHAPE, train_qs.shape[1], num_answers, layers)

  # save model every epoch
  checkpoint = keras.callbacks.ModelCheckpoint('./models/'+model_name, period=1) 

  print('fitting...')
  history = model.fit(x = my_training_batch_generator,
      steps_per_epoch = np.ceil(train_answers.shape[0] / BATCH_SIZE),
      epochs=epochs,
      verbose=1,
      validation_data = my_validation_batch_generator,
      validation_steps = np.ceil(test_answers.shape[0] / BATCH_SIZE),
      callbacks=[checkpoint]
  )

  with open('./history/{}_hist'.format(model_name), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
  file_pi.close()

  model.save('./models/'+model_name)

# #seq
# # 3 layers, yes/no (top 2)
# run_model(epochs=EPOCHS, model_name='3L_yes_no_seq', layers=3, check_top_ans=2, removeYesNo=False, get_data=seq_get_data, type='seq')

# # 5 layers, yes/no (top 2)
# run_model(epochs=EPOCHS, model_name='5L_yes_no_seq', layers=5, check_top_ans=2, removeYesNo=False, get_data=seq_get_data, type='seq')

# # 3 layers, top 10 answers (not including yes/no)
# run_model(epochs=EPOCHS, model_name='3L_top_10_seq', layers=3, check_top_ans=10, removeYesNo=True, get_data=seq_get_data, type='seq')

# # 5 layers, top 10 answers (not including yes/no)
# run_model(epochs=EPOCHS, model_name='5L_top_10_seq', layers=5, check_top_ans=10, removeYesNo=True, get_data=seq_get_data, type='seq')

# # 3 layers, top 100 answers (not including yes/no)
# run_model(epochs=EPOCHS, model_name='3L_top_100_seq', layers=3, check_top_ans=100, removeYesNo=True, get_data=seq_get_data, type='seq')

# # 5 layers, top 100 answers (not including yes/no)
# run_model(epochs=EPOCHS, model_name='5L_top_100_seq', layers=5, check_top_ans=100, removeYesNo=True, get_data=seq_get_data, type='seq')

# 3 layers, yes/no (top 2)
run_model(epochs=EPOCHS, model_name='3L_yes_no_seq_lessDO', layers=3, check_top_ans=2, removeYesNo=False, get_data=seq_get_data, type='seq')

# 5 layers, yes/no (top 2)
run_model(epochs=EPOCHS, model_name='5L_yes_no_seq_lessDO', layers=5, check_top_ans=2, removeYesNo=False, get_data=seq_get_data, type='seq')

# # 3 layers, top 10 answers (not including yes/no)
# run_model(epochs=EPOCHS, model_name='3L_top_10_seq', layers=3, check_top_ans=10, removeYesNo=True, get_data=seq_get_data, type='seq')

# # 5 layers, top 10 answers (not including yes/no)
# run_model(epochs=EPOCHS, model_name='5L_top_10_seq', layers=5, check_top_ans=10, removeYesNo=True, get_data=seq_get_data, type='seq')

# # 3 layers, top 100 answers (not including yes/no)
# run_model(epochs=EPOCHS, model_name='3L_top_100_seq', layers=3, check_top_ans=100, removeYesNo=True, get_data=seq_get_data, type='seq')

# # 5 layers, top 100 answers (not including yes/no)
# run_model(epochs=EPOCHS, model_name='5L_top_100_seq', layers=5, check_top_ans=100, removeYesNo=True, get_data=seq_get_data, type='seq')


# bow
# 3 layers, yes/no (top 2)
run_model(epochs=EPOCHS, model_name='3L_yes_no_bow', layers=3, check_top_ans=2, removeYesNo=False, get_data=bow_get_data, type='bow')

# # 5 layers, yes/no (top 2)
# run_model(epochs=EPOCHS, model_name='5L_yes_no_bow', layers=5, check_top_ans=2, removeYesNo=False, get_data=bow_get_data, type='bow')

# # 3 layers, top 10 answers (not including yes/no)
# run_model(epochs=EPOCHS, model_name='3L_top_10_bow', layers=3, check_top_ans=10, removeYesNo=True, get_data=bow_get_data, type='bow')

# # 5 layers, top 10 answers (not including yes/no)
# run_model(epochs=EPOCHS, model_name='5L_top_10_bow', layers=5, check_top_ans=10, removeYesNo=True, get_data=bow_get_data, type='bow')

# # 3 layers, top 100 answers (not including yes/no)
# run_model(epochs=EPOCHS, model_name='3L_top_100_bow', layers=3, check_top_ans=100, removeYesNo=True, get_data=bow_get_data, type='bow')

# # 5 layers, top 100 answers (not including yes/no)
# run_model(epochs=10, model_name='5L_top_100_bow', layers=5, check_top_ans=100, removeYesNo=True, get_data=bow_get_data, type='bow')
run_model(epochs=EPOCHS, model_name='3L_yes_no_dropout', layers=3, check_top_ans=2, removeYesNo=False, get_data=bow_get_data, type='bow')

# # 4 layers, yes/no (top 2)
# run_model(epochs=EPOCHS, model_name='4L_yes_no_bow', layers=4, check_top_ans=2, removeYesNo=False, get_data=bow_get_data, type='bow')

# 5 layers, yes/no (top 2)
run_model(epochs=EPOCHS, model_name='5L_yes_no_dropout', layers=5, check_top_ans=2, removeYesNo=False, get_data=bow_get_data, type='bow')

# # 3 layers, top 10 answers (not including yes/no)
# run_model(epochs=EPOCHS, model_name='3L_top_10_bow', layers=3, check_top_ans=10, removeYesNo=True, get_data=bow_get_data, type='bow')

# # 4 layers, top 10 answers (not including yes/no)
# run_model(epochs=EPOCHS, model_name='4L_top_10_bow', layers=4, check_top_ans=10, removeYesNo=True, get_data=bow_get_data, type='bow')

# # 5 layers, top 10 answers (not including yes/no)
# run_model(epochs=EPOCHS, model_name='5L_top_10_bow', layers=5, check_top_ans=10, removeYesNo=True, get_data=bow_get_data, type='bow')

# # 3 layers, top 100 answers (not including yes/no)
# run_model(epochs=EPOCHS, model_name='3L_top_100_bow', layers=3, check_top_ans=100, removeYesNo=True, get_data=bow_get_data, type='bow')

# # 4 layers, top 100 answers (not including yes/no)
# run_model(epochs=EPOCHS, model_name='4L_top_100_bow', layers=4, check_top_ans=100, removeYesNo=True, get_data=bow_get_data, type='bow')

# # 5 layers, top 100 answers (not including yes/no)
# run_model(epochs=EPOCHS, model_name='5L_top_100_bow', layers=5, check_top_ans=100, removeYesNo=True, get_data=bow_get_data, type='bow')