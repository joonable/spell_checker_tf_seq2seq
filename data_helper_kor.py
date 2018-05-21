import pandas as pd
import random
import jamo
import numpy as np

cho = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")  # len = 19
jung = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")  # len = 21
jong = list("ㄱ/ㄲ/ㄳ/ㄴ/ㄵ/ㄶ/ㄷ/ㄹ/ㄺ/ㄻ/ㄼ/ㄽ/ㄾ/ㄿ/ㅀ/ㅁ/ㅂ/ㅄ/ㅅ/ㅆ/ㅇ/ㅈ/ㅊ/ㅋ/ㅌ/ㅍ/ㅎ".split('/'))  # len = 27
number = list("0123456789")
alphabets = list("abcdefghijklmnopqrstuvwxyz")

only_cho = list("ㄸㅃㅉ")
only_jong = list("ㄳ/ㄵ/ㄶ/ㄺ/ㄻ/ㄼ/ㄽ/ㄾ/ㄿ/ㅀ/ㅄ".split('/'))
safe_cho_jong = list(set(cho + jong) - set(only_cho + only_jong))

vocabulary_list = list("SEP ") + list(set(cho + jong)) + jung + number + alphabets
vocabulary_dict = {n:i for i, n in enumerate(vocabulary_list)}

# df_train = pd.read_csv('./df_train_with_noise_kor.csv')
# df_test = pd.read_csv('./df_test_with_noise_kor.csv')
# vowels = list('aeiou')
# consonants = list('bcdfghjklmnpqrstvwxyz')

def read_file(path):
    df = pd.read_csv(path)
    return df


def preprocessing(df):
    # preprecessing
    for i in range(0, len(df)):
        y_list = list(df.loc[i, 'y'].lower())
        for j in range(0, len(y_list)):
            if y_list[j] not in vocabulary_list:
                y_list[j] = ''
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
    temp_actual = []

    for i in range(0, initial_len):
        x, y, actual = get_misspelled_words(df.loc[i, 'y'])
        temp_x += x
        temp_y += y
        temp_actual += actual
    d = {'x':temp_x, 'y':temp_y, 'actual':temp_actual}
    new_df = pd.DataFrame(d)
    return new_df


def get_misspelled_words(word):
    # splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    # word_list = "".join(word)
    misspelled_word_list = []

    # 앞의 글자 없애기
    if rand_check(5):
        rand_num = random.randint(1, 2)
        misspelled_word_list.append(word[rand_num:])

    n = 6
    if not len(word) < 6:
        # 부분적으로 글자 없애기
        misspelled_word = ''
        for i in range(int(len(word) / n)):
            rand_num = random.randint(0, n)
            idx = (i * n) + rand_num
            min_idx = (i * n)
            max_idx = ((i + 1) * n)
            if max_idx > len(word) - 1: break
            if rand_check(3):
                try:
                    if word[idx] in only_jong:
                        word[idx] = safe_cho_jong[random.randint(0, len(safe_cho_jong)) % len(safe_cho_jong)]
                    else:
                        misspelled_word = misspelled_word + word[min_idx:idx] + word[idx + 1:max_idx]
                except:
                    print(word)
            else:
                misspelled_word = misspelled_word + word[min_idx:max_idx]

        if word is not misspelled_word and word is not None:
            misspelled_word_list.append(misspelled_word)

    n = 6
    if not len(word) < n:
        # 부분적으로 글자 바꾸기
        misspelled_word = ''
        for i in range(int(len(word) / n)):
            rand_num = random.randint(0, n)
            idx = (i * n) + rand_num
            min_idx = (i * n)
            max_idx = ((i + 1) * n)
            if max_idx > len(word) - 1: break

            if max_idx > len(word): break
            if rand_check(3):
                if word[idx] in alphabets:
                    type = alphabets
                elif word[idx] in number:
                    type = number
                elif word[idx] in jung:
                    type = jung
                elif word[idx] in only_cho or safe_cho_jong:
                    type = cho
                elif word[idx] in only_jong:
                    type = jong
                else:
                    type = [" "]

                changed_char = type[random.randint(0, len(type)) % len(type)]
                misspelled_word = misspelled_word + word[min_idx:idx] + changed_char + word[idx + 1:max_idx]
            else:
                misspelled_word = misspelled_word + word[min_idx:max_idx]

        if word is not misspelled_word and word is not None:
            misspelled_word_list.append(misspelled_word)

    # 뒤의 글자 없애기
    if rand_check(7):
        rand_num = random.randint(1, 2)
        misspelled_word_list.append(word[:len(word) - rand_num])

    return misspelled_word_list, [word] * len(misspelled_word_list), [jamo.join_jamos(word)] * len(misspelled_word_list)


def cleasing_df(df):
    df = df.dropna()
    df = df.drop_duplicates()
    df = df[df['x'] != '']
    df['len'] = df.x.apply(lambda x:len(x))
    df = df[df.len > 7]
    df = df.reset_index(drop = True)
    df = df.drop(['len'], axis = 1)

    return df


def split_to_csv(df):
    df = df.sample(frac = 1).reset_index(drop = True)
    df_train = df.head(int(len(df) * 0.8))
    df_test = df.tail(len(df) - int(len(df) * 0.8))

    df_train.to_csv('./df_train_with_noise_kor.csv', index = False)
    df_test.to_csv('./df_test_with_noise_kor.csv', index = False)


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