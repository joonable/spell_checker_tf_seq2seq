import numpy as np
import pandas as pd
import tensorflow as tf


class SpellChecker():
    def __init__(self, config):
        self.lr = config['lr']
        self.n_hidden = config['n_hidden']
        self.total_epoch = config['total_epoch']
        self.char_arr = list("SEPabcdefghijklmnopqrstuvwxyz ")
        self.n_class = self.n_input = self.dic_len = len(self.char_arr)

        self.embedding_size = 4

        self.encoder_inputs = tf.placeholder(dtype = tf.float32, shape = [None, None, self.n_input],
                                             name = "encoder_inputs")
        self.decoder_inputs = tf.placeholder(dtype = tf.float32, shape = [None, None, self.n_input],
                                             name = "decoder_inputs")
        self.decoder_outputs = tf.placeholder(tf.int64, [None, None], name = "decoder_outputs")
        self.target_weights = tf.placeholder(tf.float32, [None, None], name = "target_weights")

        self.encoder_length = tf.placeholder(tf.int32, [None], name = "encoder_length")
        self.decoder_length = tf.placeholder(tf.int32, [None], name = "decoder_length")
        # Embedding
        # Look up embedding:
        #   encoder_inputs: [max_time, batch_size]
        #   encoder_emb_inp: [max_time, batch_size, embedding_size]
        # self.embedding_encoder = tf.get_variable("embedding_encoder", [self.dic_len, self.embedding_size])
        # self.encoder_emb_inp = tf.nn.embedding_lookup(self.embedding_encoder, self.encoder_inputs)
        #
        # self.embedding_decoder = tf.get_variable("embedding_decoder", [self.dic_len, self.embedding_size])
        # self.decoder_emb_inp = tf.nn.embedding_lookup(self.embedding_decoder, self.decoder_inputs)
        # self.decoder_emb_outp = tf.nn.embedding_lookup(self.embedding_decoder, self.decoder_outputs)
        #
        self.projection_layer = tf.layers.Dense(self.dic_len, use_bias = False)

        # [batch_size, time_steps, input_size]

        with tf.variable_scope('encode'):
            self.enc_cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden)
            self.enc_cell = tf.nn.rnn_cell.DropoutWrapper(self.enc_cell, output_keep_prob = 0.5)
            self.outputs, self.enc_states = tf.nn.dynamic_rnn(cell = self.enc_cell, inputs = self.encoder_inputs,
                                                              dtype = tf.float32, sequence_length = self.encoder_length)

        with tf.variable_scope('decode'):
            self.dec_cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden)
            self.dec_cell = tf.nn.rnn_cell.DropoutWrapper(self.dec_cell, output_keep_prob = 0.5)

            self.helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_inputs, self.decoder_length)
            self.decoder = tf.contrib.seq2seq.BasicDecoder(self.dec_cell, self.helper, self.enc_states,
                                                           output_layer = self.projection_layer)

            self.outputs, self.dec_states, self.final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(self.decoder)

        self.logits = self.outputs.rnn_output

        # self.logit = tf.layers.dense(self.outputs, self.n_class, activation = None)
        self.prediction = tf.argmax(self.logits, 2)
        # self.accuracy = tf.metrics.accuracy(predictions = self.prediction, labels = self.decoder_outputs)

        self.crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.logits,
                                                                       labels = self.decoder_outputs)

        self.cost = (tf.reduce_mean(self.crossent * self.target_weights))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

        self.df_train = pd.read_csv('./df_train.csv', index_col = False)
        self.df_test = pd.read_csv('./df_test.csv', index_col = False)
        self.df_train = self.df_train[['x', 'y']]
        self.df_test = self.df_test[['x', 'y']]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def make_batch(self, df):

        num_dic = {n:i for i, n in enumerate(self.char_arr)}
        dic_len = len(num_dic)

        enc_input_batch = []
        dec_input_batch = []
        dec_output_batch = []
        target_weights_batch = []

        enc_len_batch = []
        dec_len_batch = []

        enc_max_len = 0
        dec_max_len = 0

        # preprecessing
        for i in range(0, len(df)):
            df.loc[i, 'x'] = df.loc[i, 'x'].lower()
            df.loc[i, 'y'] = df.loc[i, 'y'].lower()

            if enc_max_len < len(df.loc[i, 'x']): enc_max_len = len(df.loc[i, 'x'])
            if dec_max_len < len(df.loc[i, 'y']) + 1: dec_max_len = len(df.loc[i, 'y']) + 1

            enc_len_batch.append(len(df.loc[i, 'x']))
            dec_len_batch.append(len(df.loc[i, 'y']) + 1)

            x_list = list(df.loc[i, 'x'])
            for j in range(0, len(x_list)):
                if x_list[j] not in self.char_arr:
                    x_list[j] = ' '
            df.loc[i, 'x'] = "".join(x_list)

            y_list = list(df.loc[i, 'y'])
            # y_list = list(df.loc[i, 'y'] + (52 - len(df.loc[i, 'y'])) * 'P')
            for j in range(0, len(y_list)):
                if y_list[j] not in self.char_arr:
                    y_list[j] = ' '
            df.loc[i, 'y'] = "".join(y_list)

        # target = ['P' * 51 for n in range(0, len(df))]
        for i in range(0, len(df)):
            input = [num_dic[n] for n in df.loc[i, 'x'].lower()]
            output = [num_dic[n] for n in ('S' + df.loc[i, 'y'].lower())]
            target = [num_dic[n] for n in (df.loc[i, 'y'].lower() + 'E')]

            target_weights_batch.extend([([1] * len(target)) + ([0] * (dec_max_len - len(target)))])

            input = input + [2] * (enc_max_len - len(input))
            output = output + [2] * (dec_max_len - len(output))
            target = target + [2] * (dec_max_len - len(target))

            enc_input_batch.append(np.eye(dic_len)[input])
            dec_input_batch.append(np.eye(dic_len)[output])
            dec_output_batch.append(target)

        # target_weights_batch = tf.squeeze(target_weights_batch)
        return enc_input_batch, dec_input_batch, dec_output_batch, target_weights_batch, enc_len_batch, dec_len_batch

    def train(self):
        train_enc_input_batch, train_dec_input_batch, train_dec_output_batch, train_target_weights_batch, train_enc_len_batch, train_dec_len_batch = self.make_batch(
            self.df_train)
        val_enc_input_batch, val_dec_input_batch, val_dec_output_batch, val_target_weights_batch, val_enc_len_batch, val_dec_len_batch = self.make_batch(
            self.df_test)
        for epoch in range(self.total_epoch):
            _, loss = self.sess.run([self.train_op, self.cost],
                                    feed_dict = {self.encoder_inputs:train_enc_input_batch,
                                                 self.decoder_inputs:train_dec_input_batch,
                                                 self.decoder_outputs:train_dec_output_batch,
                                                 self.target_weights:train_target_weights_batch,
                                                 self.encoder_length:train_enc_len_batch,
                                                 self.decoder_length:train_dec_len_batch})
            # val_loss = self.sess.run([self.cost],
            #                          feed_dict = {self.encoder_inputs:val_enc_input_batch,
            #                                       self.decoder_inputs:val_dec_input_batch,
            #                                       self.decoder_outputs:val_dec_output_batch,
            #                                       self.target_weights:val_target_weights_batch,
            #                                       self.encoder_length:val_enc_len_batch,
            #                                       self.decoder_length:val_dec_len_batch})
            print('Epoch:', '%04d' % (epoch + 1), 'cost = ', '{:.6f}'.format(loss))

    def test(self):
        self.enc_input_batch, self.dec_input_batch, self.dec_output_batch, self.target_weights_batch, self.enc_len_batch, self.dec_len_batch = self.make_batch(
            self.df_test)

        self.results, loss = self.sess.run([self.prediction, self.cost],
                                           feed_dict = {self.encoder_inputs:self.enc_input_batch,
                                                        self.decoder_inputs:self.dec_input_batch,
                                                        self.decoder_outputs:self.dec_output_batch,
                                                        self.target_weights:self.target_weights_batch,
                                                        self.encoder_length:self.enc_len_batch,
                                                        self.decoder_length:self.dec_len_batch})
        print('cost = ', '{:.6f}'.format(loss))

        # decoded = [char_arr[i] for i in result[0]]
        decoded = []
        for result in self.results:
            decoded.append([self.char_arr[i] for i in result])
            #
        self.translated = []
        for result in decoded:
            end = result.index('E')
            # translated.append()
            self.translated.append([''.join(result[:end])])
            # translated = []
        # for result in decoded:
        #     end = result.index('E')
        #     translated.append(''.join(result[:end]))

        return self.translated

def main():
    config = {}
    config['lr'] = 0.005
    config['n_hidden'] = 256
    config['total_epoch'] = 30
    spell_checker = SpellChecker(config)
    spell_checker.train()
    test_result = spell_checker.test()
    pd.DataFrame(test_result).to_csv('./test_result', index = False)

if __name__ == '__main__':
    main()
