import tensorflow as tf
import pandas as pd

from model import Model


class AttentionSpellChecker(Model):
    def __init__(self, config, file_name):
        super(AttentionSpellChecker, self).__init__(config = config, file_name = file_name)
        self.build_model()

    def build_model(self):
        self.build_embedding_layer()
        self.build_encoder()
        self.build_decoder()
        super(AttentionSpellChecker, self).build_model()

    def build_embedding_layer(self):
        with tf.variable_scope('embedding'):
            super(AttentionSpellChecker, self).build_embedding_layer()

    def build_encoder(self):
        with tf.variable_scope('encode'):
            super(AttentionSpellChecker, self).build_encoder()

    def build_decoder(self):
        with tf.variable_scope('attention'):
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                self.n_hidden, self.outputs, memory_sequence_length = self.encoder_length)

        with tf.variable_scope('decode'):
            super(AttentionSpellChecker, self).build_decoder()

            self.dec_cell = tf.contrib.seq2seq.AttentionWrapper(
                self.dec_cell, attention_mechanism, attention_layer_size = self.n_hidden)
            self.helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_emb_inp, self.decoder_length)
            projection_layer = tf.layers.Dense(self.dic_len, use_bias = True)
            self.decoder = tf.contrib.seq2seq.BasicDecoder(
                self.dec_cell, self.helper,
                initial_state = self.dec_cell.zero_state(self.current_batch_size, tf.float32)
                    .clone(cell_state = self.enc_states),
                output_layer = projection_layer)
            self.outputs, self.dec_states, self.final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(self.decoder)

    def save_current_session(self, current_step):
        path = self.saver.save(self.sess, self.checkpoint_prefix, global_step = current_step)
        print('Saved model {} at step {}'.format(path, self.best_at_step))

    def restore_best_session(self, best_at_step = None):
        if not best_at_step:
            best_at_step = self.best_at_step
        self.saver = tf.train.import_meta_graph("{}.meta".format(self.checkpoint_prefix + '-' + str(best_at_step)))
        self.saver.restore(self.sess, self.checkpoint_prefix + '-' + str(best_at_step))


def main():
    config = {}
    config['lr'] = 0.001
    config['n_hidden'] = 32
    config['total_epoch'] = 3
    config['batch_size'] = 256
    config['n_eval'] = 20
    config['embedding_size'] = 8
    spell_checker = AttentionSpellChecker(config, file_name = "__")

    spell_checker.train()
    spell_checker.restore_best_session()
    spell_checker.test(spell_checker.df_train, 'train_', False)
    spell_checker.test(spell_checker.df_test, 'test_', False)

if __name__ == '__main__':
    main()
