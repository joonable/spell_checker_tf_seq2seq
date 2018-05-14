import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time
import shutil


class SpellChecker():
    def __init__(self, config):
        self.lr = config['lr']
        self.n_hidden = config['n_hidden']
        self.total_epoch = config['total_epoch']
        self.batch_size = config['batch_size']
        self.embedding_size = config['embedding_size']
        self.char_arr = list("SEPabcdefghijklmnopqrstuvwxyz ")
        self.num_dic = {n:i for i, n in enumerate(self.char_arr)}
        self.n_class = self.n_input = self.dic_len = len(self.char_arr)
        self.n_eval = config['n_eval']

        self.training_mode = True
        self.output_keep_prob = tf.placeholder(tf.float32)

        # Checkpoint files will be saved in this directory during training
        self.timestamp = str(int(time.time()))
        self.checkpoint_dir = './checkpoints_' + self.timestamp + '/'
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
        os.makedirs(self.checkpoint_dir)
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, 'model')

        self.encoder_inputs = tf.placeholder(dtype = tf.int32, shape = [None, None],
                                             name = "encoder_inputs")
        self.decoder_inputs = tf.placeholder(dtype = tf.int32, shape = [None, None],
                                             name = "decoder_inputs")
        self.decoder_outputs = tf.placeholder(tf.int64, [None, None], name = "decoder_outputs")
        self.target_weights = tf.placeholder(tf.float32, [None, None], name = "target_weights")

        self.encoder_length = tf.placeholder(tf.int32, [None], name = "encoder_length")
        self.decoder_length = tf.placeholder(tf.int32, [None], name = "decoder_length")

        self.global_step = tf.Variable(0, name = 'global_step', trainable = False)

        with tf.variable_scope('embedding'):
            self.embedding = tf.get_variable("embedding_layer", [self.dic_len, self.embedding_size], trainable = True)
            self.encoder_emb_inp = tf.nn.embedding_lookup(self.embedding, self.encoder_inputs)
            self.decoder_emb_inp = tf.nn.embedding_lookup(self.embedding, self.decoder_inputs)

        with tf.variable_scope('encode'):
            self.enc_cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden)
            self.enc_cell = tf.nn.rnn_cell.DropoutWrapper(self.enc_cell, output_keep_prob = self.output_keep_prob)
            self.outputs, self.enc_states = tf.nn.dynamic_rnn(cell = self.enc_cell, inputs = self.encoder_emb_inp,
                                                              dtype = tf.float32, sequence_length = self.encoder_length)

        with tf.variable_scope('decode'):
            self.dec_cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden)
            self.dec_cell = tf.nn.rnn_cell.DropoutWrapper(self.dec_cell, output_keep_prob = self.output_keep_prob)
            self.projection_layer = tf.layers.Dense(self.dic_len, use_bias = True)

            self.helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_emb_inp, self.decoder_length)
            self.decoder = tf.contrib.seq2seq.BasicDecoder(self.dec_cell, self.helper, self.enc_states,
                                                           output_layer = self.projection_layer)

            self.outputs, self.dec_states, self.final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(self.decoder)

        with tf.variable_scope('output'):
            self.logits = self.outputs.rnn_output
            self.prediction = tf.argmax(self.logits, axis = 2)

        with tf.variable_scope('Cost'):
            self.crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.logits,
                                                                           labels = self.decoder_outputs)
            self.cost = (tf.reduce_mean(self.crossent * self.target_weights))
            tf.summary.scalar('cost', self.cost)

        with tf.variable_scope('Accuracy'):
            correct_predictions = tf.equal(self.prediction, self.decoder_outputs)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))
            tf.summary.scalar('accuracy', self.accuracy)

        with tf.variable_scope('optimiser'):
            self.params = tf.trainable_variables()
            self.gradients = tf.gradients(self.cost, self.params)
            self.clipped_gradients, _ = tf.clip_by_global_norm(self.gradients, 1)
            self.opimiser = tf.train.AdamOptimizer(self.lr)
            self.train_op = self.opimiser.apply_gradients(zip(self.clipped_gradients, self.params),
                                                          global_step = self.global_step)

        self.df_train = pd.read_csv('./df_train.csv', index_col = False)[['x', 'y']]
        self.df_test = pd.read_csv('./df_test.csv', index_col = False)[['x', 'y']]

        self.graph = tf.Graph()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.train_writer = tf.summary.FileWriter('./train', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def batch_iter(self, data, batch_size, num_epochs):
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int(data_size / batch_size) + 1

        for epoch in range(num_epochs):
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield data[start_index:end_index]

    def make_batch_emb(self, df):
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
            if enc_max_len < len(df.loc[i, 'x']): enc_max_len = len(df.loc[i, 'x'])
            if dec_max_len < len(df.loc[i, 'y']) + 1: dec_max_len = len(df.loc[i, 'y']) + 1

            enc_len_batch.append(len(df.loc[i, 'x']))
            dec_len_batch.append(len(df.loc[i, 'y']) + 1)

        for i in range(0, len(df)):
            input = [self.num_dic[n] for n in df.loc[i, 'x'].lower()]
            output = [self.num_dic[n] for n in ('S' + df.loc[i, 'y'].lower())]
            target = [self.num_dic[n] for n in (df.loc[i, 'y'].lower() + 'E')]

            target_weights_batch.extend([([1] * len(target)) + ([0] * (dec_max_len - len(target)))])

            # pad sentence with 'P'
            input = input + [2] * (enc_max_len - len(input))
            output = output + [2] * (dec_max_len - len(output))
            target = target + [2] * (dec_max_len - len(target))

            enc_input_batch.append(input)
            dec_input_batch.append(output)
            dec_output_batch.append(target)

        return enc_input_batch, dec_input_batch, dec_output_batch, target_weights_batch, enc_len_batch, dec_len_batch

    def train(self):
        x_train = self.df_train.x.tolist()
        y_train = self.df_train.y.tolist()
        train_batches = self.batch_iter(data = list(zip(x_train, y_train)), batch_size = self.batch_size,
                                        num_epochs = self.total_epoch)
        train_accuracy, train_best_accuracy, val_best_accuracy, self.best_at_step = 0, 0, 0, 0

        for train_batch in train_batches:
            current_step = tf.train.global_step(self.sess, self.global_step)
            train_enc_input_batch, train_dec_input_batch, train_dec_output_batch, train_target_weights_batch, train_enc_len_batch, train_dec_len_batch \
                = self.make_batch_emb(pd.DataFrame(train_batch, columns = ['x', 'y']))

            feed_dict = {
                self.encoder_inputs:train_enc_input_batch,
                self.decoder_inputs:train_dec_input_batch,
                self.decoder_outputs:train_dec_output_batch,
                self.target_weights:train_target_weights_batch,
                self.encoder_length:train_enc_len_batch,
                self.decoder_length:train_dec_len_batch,
                self.output_keep_prob:0.75
            }
            self.merged_summaries = tf.summary.merge_all()
            _, loss, accuracy, summary = self.sess.run([self.train_op, self.cost, self.accuracy, self.merged_summaries],
                                                       feed_dict = feed_dict)
            self.train_writer.add_summary(summary = summary, global_step = current_step)

            print('current_step = ', '{}'.format(current_step), ', cost = ', '{:.6f}'.format(loss), ', accuracy = ',
                  '{:.6f}'.format(accuracy))

            train_accuracy += accuracy

            if current_step % self.n_eval == 0:
                val_enc_input_batch, val_dec_input_batch, val_dec_output_batch, val_target_weights_batch, val_enc_len_batch, val_dec_len_batch \
                    = self.make_batch_emb(pd.DataFrame(train_batch, columns = ['x', 'y']))

                val_feed_dict = {
                    self.encoder_inputs:val_enc_input_batch,
                    self.decoder_inputs:val_dec_input_batch,
                    self.decoder_outputs:val_dec_output_batch,
                    self.target_weights:val_target_weights_batch,
                    self.encoder_length:val_enc_len_batch,
                    self.decoder_length:val_dec_len_batch,
                    self.output_keep_prob:1
                }

                val_loss, val_accuracy = self.sess.run([self.cost, self.accuracy], feed_dict = val_feed_dict)

                print('current_step = ', '{}'.format(current_step), ', val_cost = ', '{:.6f}'.format(val_loss),
                      ', val_accuracy = ', '{:.6f}'.format(val_accuracy))

                train_accuracy /= self.n_eval
                if train_accuracy > train_best_accuracy and val_accuracy > val_best_accuracy:
                    train_best_accuracy, val_best_accuracy, self.best_at_step = train_accuracy, val_accuracy, current_step

                    path = self.saver.save(self.sess, self.checkpoint_prefix, global_step = current_step)
                    print('Saved model {} at step {}'.format(path, self.best_at_step))
                    print('Best accuracy {} and {} at step {}'.format(train_best_accuracy, val_best_accuracy,
                                                                      self.best_at_step))
                train_accuracy = 0

    def test(self, df, file_name):
        self.saver.restore(self.sess, self.checkpoint_prefix + '-' + str(self.best_at_step))

        enc_input_batch, dec_input_batch, dec_output_batch, target_weights_batch, enc_len_batch, dec_len_batch \
            = self.make_batch_emb(df)

        feed_dict = {
            self.encoder_inputs:enc_input_batch,
            self.decoder_inputs:dec_input_batch,
            self.decoder_outputs:dec_output_batch,
            self.target_weights:target_weights_batch,
            self.encoder_length:enc_len_batch,
            self.decoder_length:dec_len_batch,
            self.output_keep_prob:1
        }

        results, loss, accuracy = self.sess.run([self.prediction, self.cost, self.accuracy], feed_dict = feed_dict)
        print('cost = ', '{:.6f}'.format(loss), ', accuracy = ', '{:.6f}'.format(accuracy))

        decoded_number = []
        for result in results:
            decoded_number.append([self.char_arr[i] for i in result])

        decoded_character = []
        for result in decoded_number:
            try:
                end = result.index('E')
                decoded_character.append([''.join(result[:end])])
            except:
                decoded_character.append([''.join(result)])

        pd.DataFrame(decoded_character).to_csv('./' + file_name + '_result_' + self.timestamp + '.csv', index = False)


def main():
    config = {}
    config['lr'] = 0.003
    config['n_hidden'] = 512
    config['total_epoch'] = 20
    config['batch_size'] = 256
    config['n_eval'] = 20
    config['embedding_size'] = 4
    spell_checker = SpellChecker(config)

    spell_checker.train()
    spell_checker.test(spell_checker.df_train, 'train')
    spell_checker.test(spell_checker.df_test, 'test')

if __name__ == '__main__':
    main()
