from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import json
import os
import numpy as np
import math

QUESTIONS_PATH = './data/questions.JSON'
ANSWERS_PATH = './data/answers.txt'
IMAGES_PATH = './data/images/'

# open questions
with open(QUESTIONS_PATH, 'r') as file:
    qs = json.load(file)
file.close
# go through the questions file and get the questions, answers, and ids
questions = []
ans = []
ids = []

for q in qs:
    questions.append(qs[q][0])
    ans.append(qs[q][1])
    ids.append(qs[q][2])

# open answers file and get all possible answers
all_ans = []
total_ans = 0
with open(ANSWERS_PATH, 'r') as file:
    for a in file:
        all_ans.append(a.strip())
        total_ans+=1
file.close()

# map image ids to image paths
image_paths = {}
for image in os.listdir(IMAGES_PATH):
    img_id = int(image.split('.')[0])
    image_paths[img_id] = IMAGES_PATH+str(image)

# map image ids to processed images
images = {}
for i in image_paths:
    # load image
    img = load_img(image_paths[i])
    # image to array
    img_arr = img_to_array(img)
    # normalize and save img to images object
    images[i] = img_arr/255
# get images shape 
image_shape = images[0].shape

def get_questions():
    # get all unique words in questions
    words = set('')
    for q in questions:
        # remove question mark
        q = q[:-1]
        words.update(q.split())
    num_words = len(words)

    # tokenize words
    tokens = {}
    counter = 0
    for word in words:
        tokens[word] = counter
        counter += 1

    # bag of words for questions
    # create numpy array of zeroes to hold bags of words for questions
    # size number of questions x number of total words
    questions_bow = np.zeros((len(questions),len(tokens)))
    count = 0
    for q in questions:
        # remove questions mark
        q = q[:-1]
        # turn into list
        q = q.split()
        for t in tokens:
            # check if the current word from tokens is in the question
            if t in q:
                # if it is, set np array of index count x id of t in token to 1
                questions_bow[count][tokens[t]] = 1.
        count += 1
    return questions_bow, num_words

# get questions
all_questions, num_words = get_questions()

# create model inputs
x = np.array([images[id] for id in ids])

# create model outputs
answers_idx = [all_ans.index(a) for a in ans]
y = to_categorical(answers_idx)

# set ratio for train
ratio = 0.8
# split x into train and test
ratio = math.floor(len(x)*ratio)
x_train = x[:ratio]
x_test = x[ratio:]
# split x into train and test
y_train = y[:ratio]
y_test = y[ratio:]
# split questions
train_questions = all_questions[:ratio]
test_questions = all_questions[ratio:]

def get_data():
    return (x_train, x_test, y_train, y_test, train_questions, test_questions, all_ans, num_words, image_shape)
