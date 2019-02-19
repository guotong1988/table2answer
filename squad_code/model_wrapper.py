import tensorflow as tf
import modeling

class ModelWrapper():
    def __init__(self, config, is_train):
        self.config = config
        self.header_len_placeholder = tf.placeholder(dtype=tf.int32, shape=[config["batch_size"]])
        self.context_len_placeholder = tf.placeholder(dtype=tf.int32, shape=[config["batch_size"]])
        self.question_len_placeholder = tf.placeholder(dtype=tf.int32, shape=[config["batch_size"]])
        self.input_placeholder = tf.placeholder(dtype=tf.int32, shape=[config["batch_size"],
                                                                       config["max_input_len"]])
        self.input_mask_placeholder = tf.placeholder(dtype=tf.int32, shape=[config["batch_size"],
                                                                            config["max_input_len"]])
        self.target1_placeholder = tf.placeholder(dtype=tf.int32, shape=[config["batch_size"],
                                                                         config["max_input_len"]])
        self.target2_placeholder = tf.placeholder(dtype=tf.int32, shape=[config["batch_size"],
                                                                         config["max_input_len"]])

        self.type_placeholder = tf.placeholder(dtype=tf.int32,shape=[config["batch_size"],
                                                                     config["max_input_len"]])
        main_model_config = modeling.BertConfig.from_json_file("../uncased_L-12_H-768_A-12/bert_config.json")

        main_model = modeling.BertModel(main_model_config,
                                             input_ids=self.input_placeholder,
                                             token_type_ids=self.type_placeholder,
                                             input_mask=self.input_mask_placeholder,
                                             is_training=is_train,
                                             use_one_hot_embeddings=False)

        final_hidden = main_model.get_sequence_output()#[:,:config["max_cell_num"],:]

        batch_size = config["batch_size"]
        seq_length = config["max_input_len"]
        hidden_size = main_model_config.hidden_size

        output_weights = tf.get_variable(
            "cls/squad/output_weights", [2, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "cls/squad/output_bias", [2], initializer=tf.zeros_initializer())

        final_hidden_matrix = tf.reshape(final_hidden,
                                         [batch_size * seq_length, hidden_size])
        logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        logits = tf.reshape(logits, [batch_size, seq_length, 2])
        logits = tf.transpose(logits, [2, 0, 1])
        unstacked_logits = tf.unstack(logits, axis=0)
        (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

        self.start_logits = tf.reshape(start_logits, [batch_size, seq_length])
        self.end_logits = tf.reshape(end_logits, [batch_size, seq_length])

        self.context_mask = tf.sequence_mask(self.context_len_placeholder, maxlen=config["max_input_len"], dtype=tf.float32)
        self.header_mask = tf.sequence_mask(self.header_len_placeholder,maxlen=config["max_input_len"],dtype=tf.float32)
        self.target_mask = self.context_mask - self.header_mask

        self.output_logits1 = self.start_logits * self.target_mask
        self.output_logits2 = self.end_logits * self.target_mask

        log_probs1 = tf.nn.log_softmax(self.start_logits*self.target_mask, axis=-1)
        self.loss = -tf.reduce_mean(
            tf.reduce_sum(tf.cast(self.target1_placeholder, tf.float32) * log_probs1*self.target_mask, axis=-1))

        log_probs2 = tf.nn.log_softmax(self.end_logits*self.target_mask, axis=-1)
        self.loss += -tf.reduce_mean(
            tf.reduce_sum(tf.cast(self.target2_placeholder, tf.float32) * log_probs2*self.target_mask, axis=-1))

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

