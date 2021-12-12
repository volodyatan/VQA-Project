from tensorflow.keras.utils import to_categorical
import json
import numpy as np
import math

TRAIN_AMOUNT = 60000
TEST_AMOUNT = 60000


def get_qaiapa(data, type, check_top_ans, removeYesNo):
    # # open questions
    # go through the questions file and get the questions, answers, and ids

    if check_top_ans > 0:
        # top answers
        p_ans = {}
        for q in data:
            answer = q['ans']
            if answer in p_ans:
                p_ans[answer] += 1
            else:
                p_ans[answer] = 1
        top_ans = {}
        for k, v in p_ans.items():
            top_ans[v] = top_ans.get(v, []) + [k]
        top_ans = dict(sorted(top_ans.items(),reverse=True))
        top = []
        for a in top_ans:
            if removeYesNo == True:
                if 'yes' in top_ans[a] or 'no' in top_ans[a]:
                    continue
            top += top_ans[a]
            if len(top) >= check_top_ans:
                break
        print('top ', top)
        print('top size ', len(top))

    questions = []
    ans = []
    ids = []
    all_ans = []
    image_paths = {}
    image_paths_arr = []

    if type == 'train':
        max_imgs = TRAIN_AMOUNT
    else:
        max_imgs = TEST_AMOUNT

    for q in data:
        # map image id to image path
        # get img id, img id is id of question minus last element
        # special case
        if(q['ques_id'] == 0 or q['ques_id'] == 1 or q['ques_id'] == 2):
            img_id = 0
        else:
            img_id = int(str(q['ques_id'])[:-1])
        # only use top answers
        if check_top_ans > 0:
            if not q['ans'] in top:
                continue
        # add answer to all_ans if it doesn't exist yet
        if not (q['ans'] in all_ans):
            all_ans.append(q['ans'])
        image_paths[img_id] = q['img_path']
        questions.append(q['question'])
        image_paths_arr.append(q['img_path'])
        ans.append(q['ans'])
        ids.append(img_id)
        if len(questions) >= max_imgs:
            break
    print('questions len ', len(questions))
    return questions, ans, ids, all_ans, image_paths, image_paths_arr

def get_questions(questions):
    # get all unique words in questions
    words = set('')
    max_q = 0
    for q in questions:
        # remove question mark
        q = q[:-1]
        q[0].lower()
        question = q.split()
        words.update(question)
        if len(question) > max_q:
            max_q = len(question)
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

def set_data(check_top_ans, removeYesNo):
    raw_train = json.load(open('abstract_train.json', 'r'))
    raw_test = json.load(open('abstract_test.json', 'r'))

    # train
    train_questions, train_ans, train_ids, possible_train_ans, train_img_paths, image_paths_arr = get_qaiapa(raw_train, 'train',  check_top_ans, removeYesNo)
    # test
    # test_questions, test_ans, test_ids, possible_test_ans, test_img_paths = get_qaiap(raw_test, 'test')
    # set all answers (possible answers from training set)
    all_ans = possible_train_ans
    num_ans = len(all_ans)

    # get questions bow
    # train
    train_questions_bow, num_words = get_questions(train_questions)

    # create model inputs
    x = image_paths_arr

    # create model outputs
    answers_idx = [all_ans.index(a) for a in train_ans]
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
    train_questions = train_questions_bow[:ratio]
    test_questions = train_questions_bow[ratio:]

    return x_train, x_test, y_train, y_test, train_questions, test_questions, all_ans, num_words

def bow_get_data(check_top_ans, removeYesNo):
    x_train, x_test, y_train, y_test, train_questions, test_questions, all_ans, num_words = set_data(check_top_ans, removeYesNo)
    return (x_train, x_test, y_train, y_test, train_questions, test_questions, all_ans, num_words)