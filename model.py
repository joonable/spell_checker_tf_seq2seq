import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time
import shutil
import data_helper_kor as data_helper #TODO you can choose data_helper_eng or data_helper_kor
import abc
import jamo


class Model(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, config):
        self.lr = config['lr']
        self.n_hidden = config['n_hidden']
        self.total_epoch = config['total_epoch']
        self.batch_size = config['batch_size']
        self.embedding_size = config['embedding_size']
        self.vocabulary_list = data_helper.vocabulary_list
        self.vocabulary_dict = data_helper.vocabulary_dict
        self.n_class = self.n_input = self.dic_len = len(self.vocabulary_list)
        self.n_eval = config['n_eval']
        self.training_mode = True

        self.global_step = tf.Variable(0, name = 'global_step', trainable = False)
        self.output_keep_prob = tf.placeholder(tf.float32)
        self.current_batch_size = tf.placeholder(dtype = tf.int32, shape = [], name = "current_batch_size")

        self.encoder_inputs = tf.placeholder(dtype = tf.int32, shape = [None, None], name = "encoder_inputs")
        self.decoder_inputs = tf.placeholder(dtype = tf.int32, shape = [None, None], name = "decoder_inputs")
        self.decoder_outputs = tf.placeholder(tf.int64, [None, None], name = "decoder_outputs")

    def build_model(self):
        with tf.variable_scope('output'):
            self.logits = self.outputs.rnn_output
            self.prediction = tf.argmax(self.logits, axis = 2)

        with tf.variable_scope('Cost'):
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.logits,
                                                                           labels = self.decoder_outputs)
            self.cost = (tf.reduce_mean(crossent * self.target_weights))
            tf.summary.scalar('cost', self.cost)

        with tf.variable_scope('Accuracy'):
            correct_predictions = tf.equal(self.prediction, self.decoder_outputs)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))
            tf.summary.scalar('accuracy', self.accuracy)

        with tf.variable_scope('optimiser'):
            params = tf.trainable_variables()
            gradients = tf.gradients(self.cost, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1)
            opimiser = tf.train.AdamOptimizer(self.lr)
            self.train_op = opimiser.apply_gradients(zip(clipped_gradients, params),
                                                          global_step = self.global_step)

        self.graph = tf.Graph()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.train_writer = tf.summary.FileWriter('./train', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def load_data_set(self):
        self.df_train = data_helper.df_train
        self.df_test = data_helper.df_test

    @abc.abstractmethod
    def build_embedding_layer(self):
        # with tf.variable_scope('embedding'):
        self.embedding = tf.get_variable("embedding_layer", [self.dic_len, self.embedding_size], trainable = True)
        self.encoder_emb_inp = tf.nn.embedding_lookup(self.embedding, self.encoder_inputs)
        self.decoder_emb_inp = tf.nn.embedding_lookup(self.embedding, self.decoder_inputs)

    @abc.abstractmethod
    def build_encoder(self):
        with tf.variable_scope('encode'):
            self.encoder_length = tf.placeholder(tf.int32, [None], name = "encoder_length")

            enc_cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden)
            enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob = self.output_keep_prob)
            self.outputs, self.enc_states = tf.nn.dynamic_rnn(cell = enc_cell, inputs = self.encoder_emb_inp,
                                                              dtype = tf.float32, sequence_length = self.encoder_length)

    @abc.abstractmethod
    def build_decoder(self):
        with tf.variable_scope('decode'):
            self.decoder_length = tf.placeholder(tf.int32, [None], name = "decoder_length")
            self.target_weights = tf.placeholder(tf.float32, [None, None], name = "target_weights")

            self.dec_cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden)
            self.dec_cell = tf.nn.rnn_cell.DropoutWrapper(self.dec_cell, output_keep_prob = self.output_keep_prob)

    def train(self):
        # Checkpoint files will be saved in this directory during training
        self.timestamp = str(int(time.time()))
        self.checkpoint_dir = './checkpoints_' + self.timestamp + '/'
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
        os.makedirs(self.checkpoint_dir)
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, 'model')

        self.load_data_set()
        x_train = self.df_train.x.tolist()
        y_train = self.df_train.y.tolist()
        train_batches = data_helper.batch_iter(data = list(zip(x_train, y_train)), batch_size = self.batch_size,
                                        num_epochs = self.total_epoch)
        train_accuracy, train_best_accuracy, val_best_accuracy, self.best_at_step = 0, 0, 0, 0

        for train_batch in train_batches:
            current_step = tf.train.global_step(self.sess, self.global_step)
            train_enc_input_batch, train_dec_input_batch, train_dec_output_batch, train_target_weights_batch, \
            train_enc_len_batch, train_dec_len_batch, current_batch_size_batch \
                = data_helper.make_batch(pd.DataFrame(train_batch, columns = ['x', 'y']))

            feed_dict = {
                self.encoder_inputs:train_enc_input_batch,
                self.decoder_inputs:train_dec_input_batch,
                self.decoder_outputs:train_dec_output_batch,
                self.target_weights:train_target_weights_batch,
                self.encoder_length:train_enc_len_batch,
                self.decoder_length:train_dec_len_batch,
                self.output_keep_prob:0.75,
                self.current_batch_size: current_batch_size_batch
            }
            self.merged_summaries = tf.summary.merge_all()
            _, loss, accuracy, summary = self.sess.run([self.train_op, self.cost, self.accuracy, self.merged_summaries],
                                                       feed_dict = feed_dict)
            self.train_writer.add_summary(summary = summary, global_step = current_step)

            print('current_step = ', '{}'.format(current_step), ', cost = ', '{:.6f}'.format(loss), ', accuracy = ',
                  '{:.6f}'.format(accuracy))

            train_accuracy += accuracy

            if current_step % self.n_eval == 0:
                val_enc_input_batch, val_dec_input_batch, val_dec_output_batch, \
                val_target_weights_batch, val_enc_len_batch, val_dec_len_batch, val_current_batch_size_batch  \
                    = data_helper.make_batch(self.df_test)

                val_feed_dict = {
                    self.encoder_inputs:val_enc_input_batch,
                    self.decoder_inputs:val_dec_input_batch,
                    self.decoder_outputs:val_dec_output_batch,
                    self.target_weights:val_target_weights_batch,
                    self.encoder_length:val_enc_len_batch,
                    self.decoder_length:val_dec_len_batch,
                    self.output_keep_prob:1,
                    self.current_batch_size:val_current_batch_size_batch
                }

                val_loss, val_accuracy = self.sess.run([self.cost, self.accuracy], feed_dict = val_feed_dict)
                train_accuracy /= self.n_eval
                print('current_step = ', '{}'.format(current_step), ', val_cost = ', '{:.6f}'.format(val_loss),
                      ', val_accuracy = ', '{:.6f}'.format(val_accuracy), ', train_accuracy = ', '{:.6f}'.format(train_accuracy))

                if train_accuracy > train_best_accuracy and val_accuracy > val_best_accuracy:
                    train_best_accuracy, val_best_accuracy, self.best_at_step = train_accuracy, val_accuracy, current_step

                    path = self.saver.save(self.sess, self.checkpoint_prefix, global_step = current_step)
                    print('Saved model {} at step {}'.format(path, self.best_at_step))
                    print('Best accuracy {} and {} at step {}'.format(train_best_accuracy, val_best_accuracy,
                                                                      self.best_at_step))
                train_accuracy = 0

    def test(self, df, file_name):
        self.saver.restore(self.sess, self.checkpoint_prefix + '-' + str(self.best_at_step))

        enc_input_batch, dec_input_batch, dec_output_batch, target_weights_batch, \
        enc_len_batch, dec_len_batch, current_batch_size_batch \
            = data_helper.make_batch(df)

        feed_dict = {
            self.encoder_inputs:enc_input_batch,
            self.decoder_inputs:dec_input_batch,
            self.decoder_outputs:dec_output_batch,
            self.target_weights:target_weights_batch,
            self.encoder_length:enc_len_batch,
            self.decoder_length:dec_len_batch,
            self.output_keep_prob:1,
            self.current_batch_size: current_batch_size_batch
        }

        results, loss, accuracy = self.sess.run([self.prediction, self.cost, self.accuracy], feed_dict = feed_dict)
        print('cost = ', '{:.6f}'.format(loss), ', accuracy = ', '{:.6f}'.format(accuracy))

        decoded_number = []
        for result in results:
            decoded_number.append([self.vocabulary_list[i] for i in result])

        decoded_jamo = []
        for result in decoded_number:
            try:
                end = result.index('E')
                decoded_jamo.append([''.join(result[:end])])
            except:
                decoded_jamo.append([''.join(result)])

        # df['y_char'] = df.y.apply(lambda x : jamo.join_jamos(x))
        df['predict'] = [jamo.join_jamos(x) for x in decoded_jamo]
        df.to_csv('./' + file_name + '_result_' + self.timestamp + '.csv', index = False)
