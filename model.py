import time

import tensorflow as tf
from tensorflow.python.layers import core as layers_core

FLAGS = tf.flags.FLAGS


def sample_output(embedding, output_projection=None, given_number=None):
    """Get a loop_function that extracts the previous symbol and embeds it.

    Args:
        embedding: embedding tensor for symbols.
        output_projection: None or a pair (W, B). If provided, each fed previous
        output will first be multiplied by W and added B.
        update_embedding: Boolean; if False, the gradients will not propagate
        through the embeddings.

    Returns:
        A loop function.
    """

    def loop_function(prev, _):
        prev = tf.nn.xw_plus_b(prev, output_projection[0], output_projection[1])
        prev_symbol = tf.cast(
            tf.reshape(tf.multinomial(prev, 1), [FLAGS.batch_size * FLAGS.max_dec_sen_num]),
            tf.int32)
        emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
        return emb_prev

    def loop_function_max(prev, _):
        """function that feed previous model output rather than ground truth."""
        if output_projection is not None:
            prev = tf.nn.xw_plus_b(prev, output_projection[0], output_projection[1])
        prev_symbol = tf.argmax(prev, 1)
        emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
        return emb_prev

    def loop_given_function(prev, i):
        return tf.cond(
            tf.less(i, 2), lambda: loop_function(prev, i), lambda: loop_function_max(prev, i))

    return loop_function, loop_function_max, loop_given_function


class Generator(object):
    def __init__(self, hps, vocab):
        self._hps = hps
        self._vocab = vocab

    def _add_placeholders(self):
        hps = self._hps

        if FLAGS.run_method == 'auto-encoder':
            self._enc_batch = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch')
            self._enc_lens = tf.placeholder(tf.int32, [hps.batch_size], name='enc_lens')

        self._dec_batch = tf.placeholder(
            tf.int32, [hps.batch_size, hps.max_dec_sen_num, hps.max_dec_steps], name='dec_batch')
        self._target_batch = tf.placeholder(
            tf.int32, [hps.batch_size * hps.max_dec_sen_num, hps.max_dec_steps],
            name='target_batch')
        self._dec_padding_mask = tf.placeholder(
            tf.float32, [hps.batch_size * hps.max_dec_sen_num, hps.max_dec_steps],
            name='dec_padding_mask')
        self.reward = tf.placeholder(
            tf.float32, [hps.batch_size * hps.max_dec_sen_num, hps.max_dec_steps], name='reward')
        self.dec_lens = tf.placeholder(tf.int32, [hps.batch_size], name='dec_lens')
        self.dec_sen_lens = tf.placeholder(tf.int32, [hps.batch_size], name='dec_sen_lens')

    def _make_feed_dict(self, batch, just_enc=False):
        with tf.device('/cpu:0'):
            feed_dict = {}
            if FLAGS.run_method == 'auto-encoder':
                feed_dict[self._enc_batch] = (batch.enc_batch)
                feed_dict[self._enc_lens] = (batch.enc_lens)

            feed_dict[self._dec_batch] = (batch.dec_batch)
            feed_dict[self._target_batch] = (batch.target_batch)
            feed_dict[self._dec_padding_mask] = (batch.dec_padding_mask)
            feed_dict[self.dec_lens] = (batch.dec_lens)
            feed_dict[self.dec_sen_lens] = (batch.dec_sen_lens.reshape(self._hps.batch_size))

            return feed_dict

    def _add_encoder(self, encoder_inputs, seq_len):
        with tf.variable_scope('encoder'):
            cell_fw = tf.nn.rnn_cell.LSTMCell(
                self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
            cell_bw = tf.nn.rnn_cell.LSTMCell(
                self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
            ((encoder_outputs_forward, encoder_outputs_backward),
             (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(
                 cell_fw,
                 cell_bw,
                 encoder_inputs,
                 dtype=tf.float32,
                 sequence_length=seq_len,
                 swap_memory=True)
        return fw_st, bw_st, tf.concat([encoder_outputs_forward, encoder_outputs_backward], axis=-1)

    def _add_decoder(self, loop_function, loop_function_max, loop_given_function, input,
                     attention_state, embedding):
        hps = self._hps

        input = tf.reshape(input,
                           [hps.batch_size * hps.max_dec_sen_num, hps.max_dec_steps, hps.emb_dim])
        input = tf.unstack(input, axis=1)

        cell = tf.nn.rnn_cell.LSTMCell(
            hps.hidden_dim,
            initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
            state_is_tuple=True)

        decoder_outputs_pretrain, _ = tf.contrib.legacy_seq2seq.attention_decoder(
            input, self._dec_in_state, attention_state, cell, loop_function=None)

        with tf.variable_scope(tf.get_variable_scope(), reuse=True):

            decoder_outputs_sample_generator, _ = tf.contrib.legacy_seq2seq.attention_decoder(
                input, self._dec_in_state, attention_state, cell, loop_function=loop_function)

            decoder_outputs_max_generator, _ = tf.contrib.legacy_seq2seq.attention_decoder(
                input, self._dec_in_state, attention_state, cell, loop_function=loop_function_max)

            decoder_outputs_given_sample_generator, _ = tf.contrib.legacy_seq2seq.attention_decoder(
                input, self._dec_in_state, attention_state, cell, loop_function=loop_given_function)

            decoder_outputs_pretrain = tf.stack(decoder_outputs_pretrain, axis=1)
            decoder_outputs_sample_generator = tf.stack(decoder_outputs_sample_generator, axis=1)
            decoder_outputs_max_generator = tf.stack(decoder_outputs_max_generator, axis=1)
            decoder_outputs_given_sample_generator = tf.stack(
                decoder_outputs_given_sample_generator, axis=1)

        return decoder_outputs_pretrain, decoder_outputs_sample_generator, decoder_outputs_max_generator, decoder_outputs_given_sample_generator

    def _reduce_states(self, fw_st, bw_st):
        """Add to the graph a linear layer to reduce the encoder's final FW and BW state 
        into a single initial state for the decoder. This is needed because the encoder is bidirectional but the decoder is not."""
        hidden_dim = self._hps.hidden_dim
        with tf.variable_scope('reduce_final_st'):
            # Define weights and biases to reduce the cell and reduce the state
            w_reduce_c = tf.get_variable(
                'w_reduce_c', [hidden_dim * 2, hidden_dim],
                dtype=tf.float32,
                initializer=self.trunc_norm_init)
            w_reduce_h = tf.get_variable(
                'w_reduce_h', [hidden_dim * 2, hidden_dim],
                dtype=tf.float32,
                initializer=self.trunc_norm_init)
            bias_reduce_c = tf.get_variable(
                'bias_reduce_c', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
            bias_reduce_h = tf.get_variable(
                'bias_reduce_h', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)

            # Apply linear layer
            # Concatenation of fw and bw cell
            old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c])
            # Concatenation of fw and bw state
            old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h])

            # Get new cell from old cell
            new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c)
            # Get new state from old state
            new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h)
        # Return new cell and state
        return tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)

    def _add_train_op(self):
        with tf.device("/gpu:" + str(FLAGS.gpuid)):
            loss_to_minimize = self._cost
            tvars = tf.trainable_variables()
            gradients = tf.gradients(
                loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

            # Clip the gradients
            grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)

            # Add a summary
            # tf.summary.scalar('global_norm', global_norm)

            # Apply adagrad optimizer
            self._train_op = self.optimizer.apply_gradients(
                zip(grads, tvars), global_step=self.global_step, name='train_step')

    def _add_reward_train_op(self):
        with tf.variable_scope('reward_train_optimizer'):
            loss_to_minimize = self._reward_cost
            tvars = tf.trainable_variables()
            gradients = tf.gradients(
                loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

            # Clip the gradients
            grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)
            self._train_reward_op = self.optimizer.apply_gradients(
                zip(grads, tvars), global_step=self.global_step, name='train_step')

    def _output_projection_layer(self, input):
        projection_layer = tf.layers.Dense(self._vocab.size())
        return projection_layer

    def _build_model(self):
        """Add the whole generator model."""
        hps = self._hps
        vsize = self._vocab.size()

        with tf.variable_scope('seq2seq'):
            self.rand_unif_init = tf.random_normal_initializer(
                -hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=123)
            self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)
            # Add embedding matrix (shared by the encoder and decoder inputs)
            with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
                # embedding = [50000 * 128]
                embedding = tf.get_variable(
                    'embedding', [vsize, hps.emb_dim],
                    dtype=tf.float32,
                    initializer=self.trunc_norm_init)
                emb_dec_inputs = tf.nn.embedding_lookup(embedding, self._dec_batch)
                self.emb_dec_inputs = emb_dec_inputs

                if FLAGS.run_method == 'auto-encoder':
                    emb_enc_inputs = tf.nn.embedding_lookup(embedding, self._enc_batch)
                    fw_st, bw_st, encoder_outputs_word = self._add_encoder(
                        emb_enc_inputs, self._enc_lens)

                    self._dec_in_state = self._reduce_states(fw_st, bw_st)
                    sentence_level_input = tf.reshape(
                        tf.tile(
                            tf.expand_dims(self._dec_in_state.h, axis=1),
                            [1, hps.max_dec_sen_num, 1]),
                        [hps.batch_size, hps.max_dec_sen_num, hps.hidden_dim])

                    # 給 attention model 作為 input
                    encoder_outputs_word = tf.reshape(
                        tf.tile(
                            tf.expand_dims(encoder_outputs_word, axis=1),
                            [1, hps.max_dec_sen_num, 1, 1]),
                        [hps.batch_size * hps.max_dec_sen_num, -1, hps.hidden_dim * 2])
                    self.encoder_outputs_word = encoder_outputs_word
                    sentence_level_cell = tf.nn.rnn_cell.LSTMCell(
                        hps.hidden_dim,
                        initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                        state_is_tuple=True)
                    (encoder_outputs, _) = tf.nn.dynamic_rnn(
                        sentence_level_cell,
                        sentence_level_input,
                        dtype=tf.float32,
                        sequence_length=self.dec_lens,
                        swap_memory=True)
                    # 給 decoder model 作為 initial state
                    encoder_outputs = tf.reshape(
                        encoder_outputs, [hps.batch_size * hps.max_dec_sen_num, hps.hidden_dim])
                    self._dec_in_state = tf.contrib.rnn.LSTMStateTuple(
                        encoder_outputs, encoder_outputs)

            # Add the decoder.
            with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
                projection_layer = layers_core.Dense(vsize, use_bias=False)
                # decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hps.hidden_dim)
                # Attention
                # attention_states: [batch_size, max_time, num_units]
                # encoder_outputs_word:[batch_size, max_enc_steps, num_units*2 ]
                attention_states = self.encoder_outputs_word
                tiled_sequence_length = None
                batch_size = hps.batch_size
                encoder_state = self._dec_in_state
                if (FLAGS.beamsearch == 'beamsearch_test'):
                    attention_states = tf.contrib.seq2seq.tile_batch(
                        attention_states, multiplier=FLAGS.beam_width)
                    encoder_state = tf.contrib.seq2seq.tile_batch(
                        self._dec_in_state, multiplier=FLAGS.beam_width)
                    tiled_sequence_length = tf.contrib.seq2seq.tile_batch(
                        self._enc_lens, multiplier=FLAGS.beam_width)
                    batch_size = (FLAGS.batch_size * FLAGS.beam_width)

                # Create an attention mechanism
                attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                    hps.hidden_dim,
                    memory=attention_states,
                    memory_sequence_length=tiled_sequence_length)
                decoder_cell = tf.nn.rnn_cell.LSTMCell(
                    hps.hidden_dim,
                    initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                    state_is_tuple=True)
                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                    decoder_cell, attention_mechanism, attention_layer_size=hps.hidden_dim)

                initial_state = decoder_cell.zero_state(batch_size, tf.float32)

                initial_state = initial_state.clone(cell_state=encoder_state)

                if (FLAGS.beamsearch == 'beamsearch_test'):
                    inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                        cell=decoder_cell,
                        embedding=embedding,
                        start_tokens=tf.fill([hps.batch_size], 2),
                        end_token=4,
                        initial_state=initial_state,
                        beam_width=FLAGS.beam_width,
                        output_layer=projection_layer,
                        length_penalty_weight=0.0)
                    # Dynamic decoding
                    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                        inference_decoder, maximum_iterations=40)
                    self.beamsearch_outputs = outputs.predicted_ids
                elif (FLAGS.beamsearch == 'beamsearch_train'):
                    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                        embedding, tf.fill([hps.batch_size], 2), 4)
                    inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                        decoder_cell,
                        inference_helper,
                        initial_state,
                        output_layer=projection_layer)
                    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                        inference_decoder, maximum_iterations=40)
                    self.Greedy_outputs = outputs.sample_id

                    # time_major=False的时候，inputs的shape就是[batch_size, sequence_length, embedding_size] ，time_major=True时，inputs的shape为[sequence_length, batch_size, embedding_size]
                    emb_dec_inputs = tf.reshape(
                        emb_dec_inputs,
                        [hps.batch_size * hps.max_dec_sen_num, hps.max_dec_steps, hps.emb_dim])
                    helper = tf.contrib.seq2seq.TrainingHelper(
                        emb_dec_inputs, self.dec_lens * 40, time_major=False)

                    decoder_ = tf.contrib.seq2seq.BasicDecoder(
                        decoder_cell, helper, initial_state, output_layer=projection_layer)

                    final_outputs, _final_state, _final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                        decoder=decoder_, impute_finished=False, maximum_iterations=40)

                    self.decoder_outputs_pretrain = final_outputs.rnn_output
                    loss = tf.contrib.seq2seq.sequence_loss(
                        self.decoder_outputs_pretrain,
                        self._target_batch,
                        self._dec_padding_mask,
                        average_across_timesteps=True,
                        average_across_batch=False)
                    self.loss = loss

                    reward_loss = tf.contrib.seq2seq.sequence_loss(
                        self.decoder_outputs_pretrain,
                        self._target_batch,
                        self._dec_padding_mask,
                        average_across_timesteps=False,
                        average_across_batch=False) * tf.reciprocal(self.reward)

                    # tf.summary.scalar('reward_loss', reward_loss)
                    reward_loss = tf.reshape(reward_loss, [-1])

                    # # Update the cost
                    self._cost = tf.reduce_mean(loss)

                    tf.summary.scalar('loss', self._cost)
                    self.summary = tf.summary.merge_all()

                    self._reward_cost = tf.reduce_mean(reward_loss)
                    self.optimizer = tf.train.AdagradOptimizer(
                        self._hps.lr, initial_accumulator_value=self._hps.adagrad_init_acc)

    def build_graph(self, sess):
        """Add the placeholders, model, global step, train_op and summaries to the graph"""
        with tf.device("/gpu:" + str(FLAGS.gpuid)):
            tf.logging.info('Building generator graph...')
            t0 = time.time()
            self._add_placeholders()
            self._build_model()
            self.train_writer = tf.summary.FileWriter('tensorborad/train', sess.graph)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            if (FLAGS.beamsearch == 'beamsearch_train'):
                self._add_train_op()
                self._add_reward_train_op()
            t1 = time.time()
            tf.logging.info('Time to build graph: %i seconds', t1 - t0)

    def run_pre_train_step(self, sess, batch):
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            'train_op': self._train_op,
            'loss': self._cost,
            'global_step': self.global_step,
            'summary': self.summary
        }
        result = sess.run(to_return, feed_dict)
        self.train_writer.add_summary(result['summary'], result['global_step'])
        return sess.run(to_return, feed_dict)

    def run_eval_given_step(self, sess, batch):
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            'generated': self._sample_given_best_output,
        }
        return sess.run(to_return, feed_dict)

    def run_train_step(self, sess, batch, reward):
        feed_dict = self._make_feed_dict(batch)
        feed_dict[self.reward] = reward
        to_return = {
            'train_op': self._train_reward_op,
            'loss': self._reward_cost,
            'global_step': self.global_step,
        }
        return sess.run(to_return, feed_dict)

    def sample_generator(self, sess, batch):
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            # 'generated': self._sample_best_output,
            'encoder_outputs_word': self.encoder_outputs_word,
            'dec_in_state': self._dec_in_state,
            'dec_lens': self.dec_lens,
            'dec_sen_lens': self.dec_sen_lens,
            'emb_dec_inputs': self.emb_dec_inputs,
            'decoder_outputs_pretrain': self.decoder_outputs_pretrain,
            'loss': self._cost,
            'self.beamsearch_outputs': self.beamsearch_outputs
        }
        return sess.run(to_return, feed_dict)

    def max_generator(self, sess, batch):
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            'generated': self._max_best_output,
        }
        return sess.run(to_return, feed_dict)

    def run_test_language_model(self, sess, enc_batch, enc_lens, dec_batch, dec_lens):
        feed_dict = {}
        if FLAGS.run_method == 'auto-encoder':
            feed_dict[self._enc_batch] = (enc_batch)
            feed_dict[self._enc_lens] = (enc_lens)

        feed_dict[self._dec_batch] = (dec_batch)
        feed_dict[self.dec_lens] = (dec_lens)

        to_return = {
            'generated': self.Greedy_outputs,
        }
        return sess.run(to_return, feed_dict)

    def run_test_beamsearch_example(self, sess, enc_batch, enc_lens, dec_batch, dec_lens):
        feed_dict = {}
        if FLAGS.run_method == 'auto-encoder':
            feed_dict[self._enc_batch] = (enc_batch)
            feed_dict[self._enc_lens] = (enc_lens)

        feed_dict[self._dec_batch] = (dec_batch)
        feed_dict[self.dec_lens] = (dec_lens)

        to_return = {
            'beamsearch_outputs': self.beamsearch_outputs,
        }
        return sess.run(to_return, feed_dict)

    def beam_search_example(self, sess, batch):
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            'beamsearch_outputs': self.beamsearch_outputs,
        }
        return sess.run(to_return, feed_dict)

    def greedy_example(self, sess, batch):
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            'Greedy_outputs': self.Greedy_outputs,
        }
        return sess.run(to_return, feed_dict)
