import tensorflow as tf
import matrix.main_modeling as main_modeling

class ModelWrapper():
    def __init__(self, config, is_train):
        self.config = config
        self.header_len_placeholder = tf.placeholder(dtype=tf.int32, shape=[config["batch_size"]])
        self.all_cell_len_placeholder = tf.placeholder(dtype=tf.int32, shape=[config["batch_size"]])
        self.question_len_placeholder = tf.placeholder(dtype=tf.int32, shape=[config["batch_size"]])
        self.matrix_placeholder = tf.placeholder(dtype=tf.int32,shape=[config["batch_size"], config["max_cell_num"], config["max_word_num"]])
        self.input_mask_placeholder = tf.placeholder(dtype=tf.int32, shape=[config["batch_size"] , config["max_cell_num"] + config["max_question_len"]])
        self.target_placeholder = tf.placeholder(dtype=tf.int32,shape=[config["batch_size"], config["max_cell_num"]+config["max_question_len"]])
        self.question_placeholder = tf.placeholder(dtype=tf.int32,shape=[config["batch_size"], config["max_question_len"]])
        self.type_placeholder = tf.placeholder(dtype=tf.int32,shape=[config["batch_size"], config["max_cell_num"] + config["max_question_len"]])
        main_model_config = main_modeling.BertConfig.from_json_file("../uncased_L-12_H-768_A-12/bert_config.json")

        main_model = main_modeling.BertModel(main_model_config,
                                             config,
                                             input_matrix=self.matrix_placeholder,
                                             input_question=self.question_placeholder,
                                             token_type_ids=self.type_placeholder,
                                             input_mask=self.input_mask_placeholder,
                                             is_training=is_train,
                                             use_one_hot_embeddings=False)

        final_hidden = main_model.get_sequence_output()#[:,:config["max_cell_num"],:]

        """
        logits = []
        init_scale = 0.01
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        W_projection_slot = tf.get_variable("W_projection_slot", shape=[200, 2],
                                            initializer=initializer)  # [embed_size,label_size]
        b_projection_slot = tf.get_variable("b_projection_slot", shape=[2])
        dense = tf.layers.Dense(200, activation=tf.nn.tanh)
        for i in range(config["max_cell_num"]):
            feature = final_hidden[:, i, :]  # [none,self.hidden_size*2]
            hidden_states = dense(feature)  # [none,hidden_size]
            output = tf.matmul(hidden_states, W_projection_slot) + b_projection_slot  # [none,slots_num_classes]
            logits.append(output)
        # logits is a list. each element is:[none,slots_num_classes]
        logits = tf.stack(logits, axis=1)  # [none,sequence_length,slots_num_classes]
        self.predictions_slots = tf.argmax(logits, axis=2, name="predictions_slots")
        correct_prediction_slot = tf.equal(tf.cast(self.predictions_slots, tf.int32),
                                           self.target_placeholder)  # [batch_size, self.sequence_length]
        accuracy_slot = tf.reduce_mean(tf.cast(correct_prediction_slot, tf.float32), name="accuracy_slot")  # shape=()
        loss_slot = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target_placeholder, logits=logits)
        self.loss = tf.reduce_mean(loss_slot)
        """

        batch_size = config["batch_size"]
        seq_length = config["max_cell_num"]+config["max_question_len"]
        hidden_size = main_model_config.hidden_size

        output_weights = tf.get_variable(
            "cls/squad/output_weights", [1, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "cls/squad/output_bias", [1], initializer=tf.zeros_initializer())

        final_hidden_matrix = tf.reshape(final_hidden,
                                         [batch_size * seq_length, hidden_size])
        logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

        self.start_logits = tf.reshape(logits, [batch_size, seq_length])

        self.all_cell_mask = tf.sequence_mask(self.all_cell_len_placeholder, maxlen=config["max_cell_num"] + config["max_question_len"], dtype=tf.float32)
        self.header_mask = tf.sequence_mask(self.header_len_placeholder,maxlen=config["max_cell_num"]+config["max_question_len"],dtype=tf.float32)
        self.target_mask = self.all_cell_mask - self.header_mask

        self.output_logits = self.start_logits * self.target_mask

        log_probs = tf.nn.log_softmax(self.start_logits*self.target_mask, axis=-1)
        self.loss = -tf.reduce_mean(
            tf.reduce_sum(tf.cast(self.target_placeholder,tf.float32) * log_probs*self.target_mask, axis=-1))

        if not is_train: return
        optimizer = tf.train.AdamOptimizer(learning_rate=config["learning_rate"])

        # tvars = tf.trainable_variables()
        # l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
        # self.loss = self.loss + 0.01 * l2_loss
        # self.train_op = optimizer.minimize(self.loss)

        def var_filter(var_list, last_layers):
            filter_keywords = ['layer_11', 'layer_10', 'layer_9', 'layer_8']
            for var in var_list:
                if "bert" not in var.name:
                    yield var
                else:
                    for layer in last_layers:
                       kw = filter_keywords[layer]
                       if kw in var.name:
                           yield var

        def compute_gradients(tensor, var_list):
            grads = tf.gradients(tensor, var_list)
            return [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(var_list, grads)]

        tvars = list(var_filter(tf.trainable_variables(), last_layers=range(3)))
        grads = compute_gradients(self.loss, tvars)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

