## For Abstract scenes dataset from https://visualqa.org/download.html
## Help with preprocessing and extraction from https://github.com/GT-Vision-Lab/VQA_LSTM_CNN
import json

def main():

    train = []
    test = []
    imdir='data/%s/abstract_v002_%s_%012d.png'

    print('Loading annotations and questions...')
    train_annotations = json.load(open('data/training_annotations/abstract_v002_train2015_annotations.json', 'r'))
    val_annotations = json.load(open('data/validation_annotations/abstract_v002_val2015_annotations.json', 'r'))

    train_qs = json.load(open('data/training_questions/MultipleChoice_abstract_v002_train2015_questions.json', 'r'))
    val_qs = json.load(open('data/validation_questions/MultipleChoice_abstract_v002_val2015_questions.json', 'r'))
    
    ## get training data
    image_folder = 'training_images'
    image_set = 'train2015'
    for i in range(len(train_annotations['annotations'])):
        ans = train_annotations['annotations'][i]['multiple_choice_answer']
        question_id = train_annotations['annotations'][i]['question_id']
        image_id = train_annotations['annotations'][i]['image_id']
        image_path = imdir%(image_folder, image_set, train_annotations['annotations'][i]['image_id'])
        question = train_qs['questions'][i]['question']
        mc_ans = train_qs['questions'][i]['multiple_choices']

        train.append({'ques_id': question_id, 'image_id': image_id, 'img_path': image_path, 'question': question, 'MC_ans': mc_ans, 'ans': ans})

    
    ## get validation data and add to training data
    image_folder = 'validation_images'
    image_set = 'val2015'
    for i in range(len(val_annotations['annotations'])):
        ans = val_annotations['annotations'][i]['multiple_choice_answer']
        question_id = val_annotations['annotations'][i]['question_id']
        image_id = val_annotations['annotations'][i]['image_id']
        image_path = imdir%(image_folder, image_set, val_annotations['annotations'][i]['image_id'])
        question = val_qs['questions'][i]['question']
        mc_ans = val_qs['questions'][i]['multiple_choices']

        test.append({'ques_id': question_id, 'image_id': image_id, 'img_path': image_path, 'question': question, 'MC_ans': mc_ans, 'ans': ans})

    print('Training sample %d, Testing sample %d...' %(len(train), len(test)))

    json.dump(train, open('abstract_train.json', 'w'))
    json.dump(test, open('abstract_test.json', 'w'))

if __name__ == "__main__":
    main()