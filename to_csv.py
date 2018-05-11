import pandas as pd

char_arr = list("SEPabcdefghijklmnopqrstuvwxyz ")

with open('./common_typo_errors.txt', 'r+') as file:
    x_data = pd.Series([x[:x.index('-')] for x in file.readlines()])

with open('./common_typo_errors.txt', 'r+') as file:
    y_data  = pd.Series([x[x.index('-')+2:-1] for x in file.readlines()])

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
        if x_list[j] not in char_arr:
            x_list[j] = ' '
    df.loc[i, 'x'] = "".join(x_list)

    y_list = list(df.loc[i, 'y'])
    for j in range(0, len(y_list)):
        if y_list[j] not in char_arr:
            y_list[j] = ' '
    df.loc[i, 'y'] = "".join(y_list)

df = df.sample(frac = 1).reset_index(drop = True)

df_train = df.head(int(len(df) * 0.8))
df_test = df.tail(len(df) - int(len(df) * 0.8))

df_train.to_csv('./df_train.csv')
df_test.to_csv('./df_test.csv')
