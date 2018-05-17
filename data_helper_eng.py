import pandas as pd
import random
import numpy as np
from collections import Counter
import itertools

vocabulary_list = list("SEPabcdefghijklmnopqrstuvwxyz ")
vocabulary_dict = {n:i for i, n in enumerate(vocabulary_list)}

alphabets = list("abcdefghijklmnopqrstuvwxyz")
vowels = list('aeiou')
consonants = list('bcdfghjklmnpqrstvwxyz')

def build_vocab(sentences):
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_list = [word[0] for word in word_counts.most_common()]
    vocabulary_dict = {word:index for index, word in enumerate(vocabulary_list)}
    return vocabulary_dict, vocabulary_list

def preprocessing(x_data, y_data):
    #remove other words
    for i in range(0, len(y_data)):
        y_list = list(y_data[i])
        if ',' in y_list:
            y_list = y_list[:y_list.index(',')]
            y_data[i] = "".join(y_list)

    df = pd.concat([x_data, y_data], axis = 1)
    df.columns = ['x', 'y']

    # preprecessing
    for i in range(0, len(df)):
        df.loc[i, 'x'] = df.loc[i, 'x'].lower()
        df.loc[i, 'y'] = df.loc[i, 'y'].lower()

        x_list = list(df.loc[i, 'x'])
        for j in range(0, len(x_list)):
            if x_list[j] not in vocabulary_list:
                x_list[j] = ' '
        df.loc[i, 'x'] = "".join(x_list)

        y_list = list(df.loc[i, 'y'])
        for j in range(0, len(y_list)):
            if y_list[j] not in vocabulary_list:
                y_list[j] = ' '
        df.loc[i, 'y'] = "".join(y_list)
    return df

def rand_check(n):
    rand_n = random.randint(1, n)
    return rand_n % n == 0

def add_noise(df):
    # df_temp = pd.DataFrame()
    # df_temp2 = pd.DataFrame()
    # df_temp2
    initial_len = len(df)
    temp_x = []
    temp_y = []
    for i in range(0, initial_len):
        x, y = get_misspelled_words(df.loc[i, 'y'])
        temp_x += x
        temp_y += y
    d = {'x' : temp_x, 'y' : temp_y}
    temp = pd.DataFrame(d)
    df = df.append(temp).reset_index(drop = True)
    # df = pd.concat([df, temp], axis = 1).reset_index(drop = True)
    return df

def get_misspelled_words(word):
    # splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    # word_list = "".join(word)
    misspelled_word_list = []

    # 앞의 글자 없애기
    if rand_check(4):
        rand_num = random.randint(1, 2)
        misspelled_word_list.append(word[rand_num:])

    # 부분적으로 글자 없애기
    misspelled_word = ''
    for i in range(int(len(word)/4)):
        rand_num = random.randint(0, 4)
        idx = (i * 4) + rand_num
        min_idx = (i * 4)
        max_idx = ((i + 1) * 4)
        if rand_check(2):
            misspelled_word = misspelled_word + word[min_idx :idx] + word[idx+1:max_idx]
        else:
            misspelled_word = misspelled_word + word[min_idx:max_idx]

    if word is not misspelled_word and word is not None:
        misspelled_word_list.append(misspelled_word)

    # 부분적으로 글자 바꾸기
    misspelled_word = ''
    for i in range(int(len(word)/4)):
        rand_num = random.randint(0, 4)
        idx = (i * 4) + rand_num
        min_idx = (i * 4)
        max_idx = ((i + 1) * 4)
        if rand_check(2):
            misspelled_word = misspelled_word + word[min_idx :idx] + alphabets[random.randint(0, len(alphabets)) % len(alphabets)] + word[idx+1:max_idx]
        else:
            misspelled_word = misspelled_word + word[min_idx:max_idx]

    if word is not misspelled_word and word is not None:
        misspelled_word_list.append(misspelled_word)

    # 뒤의 글자 없애기
    if rand_check(4):
        rand_num = random.randint(1, 2)
        misspelled_word_list.append(word[:len(word)-rand_num])

    # df_misspelled_words = pd.DataFrame(misspelled_word_list)
    # df_misspelled_words = pd.concat([df_misspelled_words, pd.DataFrame([word] * len(misspelled_word_list))], axis = 1)
    # df_misspelled_words.columns = ['x', 'y']
    # df_misspelled_words = pd.concat([misspelled_word_list, [word] * len(misspelled_word_list)], axis = 1)
    # df_misspelled_words.columns = ['x', 'y']

    return misspelled_word_list, [word] * len(misspelled_word_list)

def cleasing_df(df):
    df = df.dropna()
    df = df.drop_duplicates()
    df = df[df['x'] != '']
    df['len'] = df.x.apply(lambda x:len(x))
    df = df[df.len > 4]
    df = df.reset_index(drop = True)
    df = df.drop(['len'], axis = 1)

    return df

def split_to_csv(df):
    df = df.sample(frac = 1).reset_index(drop = True)
    df_train = df.head(int(len(df) * 0.8))
    df_test = df.tail(len(df) - int(len(df) * 0.8))

    df_train.to_csv('./df_train_with_noise.csv')
    df_test.to_csv('./df_test_with_noise.csv')

def batch_iter(data, batch_size, num_epochs):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size / batch_size) + 1

    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield data[start_index:end_index]

def make_batch(df):
    enc_input_batch = []
    dec_input_batch = []
    dec_output_batch = []
    target_weights_batch = []

    enc_len_batch = []
    dec_len_batch = []

    enc_max_len = 0
    dec_max_len = 0
    current_batch_size = len(df)

    # preprecessing
    for i in range(0, len(df)):
        if enc_max_len < len(df.loc[i, 'x']): enc_max_len = len(df.loc[i, 'x'])
        if dec_max_len < len(df.loc[i, 'y']) + 1: dec_max_len = len(df.loc[i, 'y']) + 1

        enc_len_batch.append(len(df.loc[i, 'x']))
        dec_len_batch.append(len(df.loc[i, 'y']) + 1)

    for i in range(0, len(df)):
        input = [vocabulary_dict[n] for n in df.loc[i, 'x'].lower()]
        output = [vocabulary_dict[n] for n in ('S' + df.loc[i, 'y'].lower())]
        target = [vocabulary_dict[n] for n in (df.loc[i, 'y'].lower() + 'E')]

        target_weights_batch.extend([([1] * len(target)) + ([0] * (dec_max_len - len(target)))])

        # pad sentence with 'P'
        input = input + [2] * (enc_max_len - len(input))
        output = output + [2] * (dec_max_len - len(output))
        target = target + [2] * (dec_max_len - len(target))

        enc_input_batch.append(input)
        dec_input_batch.append(output)
        dec_output_batch.append(target)

    return enc_input_batch, dec_input_batch, dec_output_batch, target_weights_batch, \
           enc_len_batch, dec_len_batch, current_batch_size


def main():
    with open('./common_typo_errors.txt', 'r+') as file:
        x_data = pd.Series([x[:x.index('-')] for x in file.readlines()])

    with open('./common_typo_errors.txt', 'r+') as file:
        y_data  = pd.Series([x[x.index('-')+2:-1] for x in file.readlines()])

    df = preprocessing(x_data = x_data, y_data = y_data)

    df = add_noise(df = df)
    df = cleasing_df(df = df)

    split_to_csv(df = df)