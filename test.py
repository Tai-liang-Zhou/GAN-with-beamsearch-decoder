import jieba
import codecs
from pyltp import Segmentor
import opencc

t2s = opencc.OpenCC('t2s')
#a = cc.convert('五千萬')
#
s2t = opencc.OpenCC('s2t')
#b = tt.convert(a)



segmentor = Segmentor()
segmentor.load("./ltp_data_v3.4.0/cws.model")
#jieba.load_userdict('dir.txt')
words = []
with codecs.open('review_generation_dataset/train/new_train_data.csv', 'r', 'utf-8') as ask_f:
    for line in ask_f:
        line = line.split(",")
        line0 = segmentor.segment(t2s.convert(line[0]))
        words.append(("|".join(line0)))
        line1 = segmentor.segment(t2s.convert(line[1]))
        words.append(("|".join(line1)))



        
vocab = []
with codecs.open('review_generation_dataset/new_dir.txt', "r", "utf-8") as voc_f:
    for word in voc_f:
        vocab.append(word)




for word in words:
    word = word.split("|")
    for index in range(len(word)):
        if word[index] not in vocab:
            vocab.append(word[index])

with codecs.open('new_vocab.txt', "a", "utf-8") as voc_f:
    for word in vocab:
        voc_f.write(s2t.convert(word) + "\n")
    
        
        
    
        print("0 :" + line[0] + "\n")
        line0 = segmentor.segment(line[0])
        words.append(("|".join(line0)))
        print ("|".join(line0))
        print("1 :" + line[1])
        line1 = segmentor.segment(line[1])
        print ("|".join(line1))
        # sentence = jieba.cut(line[0])
        # sentence = (" ".join(sentence))
        # sentence2 = jieba.cut(line[1])
        # sentence2 = (" ".join(sentence2))
        # print (sentence+','+ sentence2)
        # wrrit_data = sentence+','+ sentence2
        # write_positive_file = codecs.open('review_generation_dataset/train/123.csv', "a", "utf-8")
        # write_positive_file.write(wrrit_data)
#        print(line)

a = words[1]
b = a.split("|")
for word in range(len(b)):
    print(b[word])

with codecs.open('review_generation_dataset/train/123.csv', 'r', 'utf-8') as ask_f:
    a = []
    for line in ask_f:
        line = line.split(",")
        for index in range(len(line)):
                    line[index] = line[index].strip()
                    line[index] = line[index].strip('\ufeff')
        a.append(line)
        print(line)




#=================================================================================================================
#新資料斷詞
words = []
with codecs.open('new_data/train_data.csv', 'r', 'utf-8') as ask_f:
    for line in ask_f:
        line = line.split(",")
        line0 = segmentor.segment(t2s.convert(line[0]))
#        words.append(("|".join(line0)))
        line1 = segmentor.segment(t2s.convert(line[1]))
#        words.append(("|".join(line1)))
        words.append((" ".join(line0))+','+(" ".join(line1))+'\n')
        
        

        
with codecs.open('new_data/student_collect_data.csv', 'a', 'utf-8') as file:
    for index in range(len(words)):
        file.write(s2t.convert(words[index]))

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#擴增詞庫
#66100
vocab = []
with codecs.open('review_generation_dataset/new_dir.txt', "r", "utf-8") as voc_f:
    for word in voc_f:
        word = t2s.convert(word)
        vocab.append(word.strip('\n'))



words = []
with codecs.open('new_data/train_data.csv', 'r', 'utf-8') as ask_f:
    for line in ask_f:
        line = line.split(",")
        line0 = segmentor.segment(t2s.convert(line[0]))
        words.append(("|".join(line0)))
        line1 = segmentor.segment(t2s.convert(line[1]))
        words.append(("|".join(line1)))
 



new_word = []       
for word in words:
    word = word.split("|")
    for index in range(len(word)):
        if word[index] not in vocab:
            new_word.append(word[index])
            vocab.append(word[index].strip('\n'))
            
with codecs.open('new_data/new_vocab.txt', "a", "utf-8") as voc_f:
    for word in vocab:
        voc_f.write(s2t.convert(word) + '\n')
#        words.append((" ".join(line0))+','+(" ".join(line1))+'\n')



tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=beam_width)
tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(encoder_final_state, multiplier=beam_width)
tiled_sequence_length = tf.contrib.seq2seq.tile_batch(sequence_length, multiplier=beam_width)
attention_mechanism = MyFavoriteAttentionMechanism(
    num_units=attention_depth,
    memory=tiled_inputs,
    memory_sequence_length=tiled_sequence_length)
attention_cell = AttentionWrapper(cell, attention_mechanism, ...)
decoder_initial_state = attention_cell.zero_state(dtype, batch_size=true_batch_size * beam_width)
decoder_initial_state = decoder_initial_state.clone(cell_state=tiled_encoder_final_state)








FinalBeamSearchDecoderOutput(
    predicted_ids=array([[[33552, 44896, 17438],[17438, 17438, 17438]]], dtype=int32), 
    beam_search_decoder_output=BeamSearchDecoderOutput(scores=array([[[-11.020242, -11.021891, -11.022562],[-21.999887, -22.001537, -22.002197]]], dtype=float32), 
    predicted_ids=array([[[33552, 44896, 17438],[17438, 17438, 17438]]], dtype=int32), 
    parent_ids=array([[[0, 0, 0],[0, 1, 2]]], dtype=int32)))










if initial_state_attention:
      attns = attention(initial_state)
for i, inp in enumerate(decoder_inputs):
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
      # If loop_function is set, we use it instead of decoder_inputs.
      if loop_function is not None and prev is not None:
        with variable_scope.variable_scope("loop_function", reuse=True):
          inp = loop_function(prev, i)
      # Merge input and previous attentions into one vector of the right size.
      input_size = inp.get_shape().with_rank(2)[1]
      if input_size.value is None:
        raise ValueError("Could not infer input size from input: %s" % inp.name)

      inputs = [inp] + attns
      x = Linear(inputs, input_size, True)(inputs)
      # Run the RNN.
      cell_output, state = cell(x, state)
      # Run the attention mechanism.
      if i == 0 and initial_state_attention:
        with variable_scope.variable_scope(
            variable_scope.get_variable_scope(), reuse=True):
          attns = attention(state)
      else:
        attns = attention(state)

      with variable_scope.variable_scope("AttnOutputProjection"):
        inputs = [cell_output] + attns
        output = Linear(inputs, output_size, True)(inputs)
      if loop_function is not None:
        prev = output
      outputs.append(output)






 with tf.variable_scope('decoder'):
            encoder_inputs_length = self.encoder_inputs_length
            # if self.beam_search:
            #     # 如果使用beam_search，则需要将encoder的输出进行tile_batch，其实就是复制beam_size份。
            #     print("use beamsearch decoding..")
            #     encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=self.beam_size)
            #     encoder_state = nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s, self.beam_size), encoder_state)
            #     encoder_inputs_length = tf.contrib.seq2seq.tile_batch(self.encoder_inputs_length, multiplier=self.beam_size)


            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=self.rnn_size, memory=encoder_outputs,
                                                                     memory_sequence_length=encoder_inputs_length)
            #attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.rnn_size, memory=encoder_outputs, memory_sequence_length=encoder_inputs_length)
            # 定义decoder阶段要是用的LSTMCell，然后为其封装attention wrapper
            decoder_cell = self._create_rnn_cell()
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell, attention_mechanism=attention_mechanism,
                                                               attention_layer_size=self.rnn_size, name='Attention_Wrapper')
            #如果使用beam_seach则batch_size = self.batch_size * self.beam_size。因为之前已经复制过一次
            #batch_size = self.batch_size if not self.beam_search else self.batch_size * self.beam_size
            batch_size = self.batch_size
            #定义decoder阶段的初始化状态，直接使用encoder阶段的最后一个隐层状态进行赋值
            decoder_initial_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32).clone(cell_state=encoder_state)
            output_layer = tf.layers.Dense(self.vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

            if self.mode == 'train':
                # 定义decoder阶段的输入，其实就是在decoder的target开始处添加一个<go>,并删除结尾处的<end>,并进行embedding。
                # decoder_inputs_embedded的shape为[batch_size, decoder_targets_length, embedding_size]
                ending = tf.strided_slice(self.decoder_targets, [0, 0], [self.batch_size, -1], [1, 1])
                decoder_input = tf.concat([tf.fill([self.batch_size, 1], self.word_to_idx['<go>']), ending], 1)
                decoder_inputs_embedded = tf.nn.embedding_lookup(embedding, decoder_input)
                #训练阶段，使用TrainingHelper+BasicDecoder的组合，这一般是固定的，当然也可以自己定义Helper类，实现自己的功能
                training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_inputs_embedded,
                                                                    sequence_length=self.decoder_targets_length,
                                                                    time_major=False, name='training_helper')
                training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=training_helper,
                                                                   initial_state=decoder_initial_state, output_layer=output_layer)
                #调用dynamic_decode进行解码，decoder_outputs是一个namedtuple，里面包含两项(rnn_outputs, sample_id)
                # rnn_output: [batch_size, decoder_targets_length, vocab_size]，保存decode每个时刻每个单词的概率，可以用来计算loss
                # sample_id: [batch_size], tf.int32，保存最终的编码结果。可以表示最后的答案
                decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                                                                          impute_finished=True,
                                                                    maximum_iterations=self.max_target_sequence_length)
                # 根据输出计算loss和梯度，并定义进行更新的AdamOptimizer和train_op
                self.decoder_logits_train = tf.identity(decoder_outputs.rnn_output)
                self.decoder_predict_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_pred_train')
                # 使用sequence_loss计算loss，这里需要传入之前定义的mask标志
                self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.decoder_logits_train,
                                                             targets=self.decoder_targets, weights=self.mask)

                # Training summary for the current batch_loss
                tf.summary.scalar('loss', self.loss)
                self.summary_op = tf.summary.merge_all()

                optimizer = tf.train.AdamOptimizer(self.learing_rate)
                trainable_params = tf.trainable_variables()
                gradients = tf.gradients(self.loss, trainable_params)
                clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
                self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))
            elif self.mode == 'decode':
                start_tokens = tf.ones([self.batch_size, ], tf.int32) * self.word_to_idx['<go>']
                end_token = self.word_to_idx['<eos>']
                # decoder阶段根据是否使用beam_search决定不同的组合，
                # 如果使用则直接调用BeamSearchDecoder（里面已经实现了helper类）
                # 如果不使用则调用GreedyEmbeddingHelper+BasicDecoder的组合进行贪婪式解码
                if self.beam_search:
                    inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell, embedding=embedding,
                                                                             start_tokens=start_tokens, end_token=end_token,
                                                                             initial_state=decoder_initial_state,
                                                                             beam_width=self.beam_size,
                                                                             output_layer=output_layer)
                else:
                    decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embedding,
                                                                               start_tokens=start_tokens, end_token=end_token)
                    inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=decoding_helper,
                                                                        initial_state=decoder_initial_state,
                                                                        output_layer=output_layer)
                decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder,
                                                                maximum_iterations=10)
                # 调用dynamic_decode进行解码，decoder_outputs是一个namedtuple，
                # 对于不使用beam_search的时候，它里面包含两项(rnn_outputs, sample_id)
                # rnn_output: [batch_size, decoder_targets_length, vocab_size]
                # sample_id: [batch_size, decoder_targets_length], tf.int32

                # 对于使用beam_search的时候，它里面包含两项(predicted_ids, beam_search_decoder_output)
                # predicted_ids: [batch_size, decoder_targets_length, beam_size],保存输出结果
                # beam_search_decoder_output: BeamSearchDecoderOutput instance namedtuple(scores, predicted_ids, parent_ids)
                # 所以对应只需要返回predicted_ids或者sample_id即可翻译成最终的结果
                if self.beam_search:
                    self.decoder_predict_decode = decoder_outputs.predicted_ids
                else:
                    self.decoder_predict_decode = tf.expand_dims(decoder_outputs.sample_id, -1)