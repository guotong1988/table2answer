import tensorflow as tf
import time
import text.data_util as data_util
import text.model_wrapper as model_wrapper
import numpy as np
import tokenization
config = {}
config["batch_size"] = 16
config["max_input_len"] = 256
# config["max_word_num"] = 10 # max word num in a cell
# config["max_question_len"] = 30
config["debug"] = False
config["learning_rate"] = 5e-5

tokenizer = tokenization.FullTokenizer(vocab_file="../uncased_L-12_H-768_A-12/vocab.txt")

valid_data_util = data_util.DataUtil_bert(json_path="../squad_data/my_valid.json",config=config,tokenizer=tokenizer)
train_data_util = data_util.DataUtil_bert(json_path="../squad_data/my_train.json",config=config,tokenizer=tokenizer)

with tf.variable_scope("Model", reuse=False):
    train_model = model_wrapper.ModelWrapper(config, is_train = True)

with tf.variable_scope("Model", reuse=True):
    valid_model = model_wrapper.ModelWrapper(config, is_train = False)


sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
# import os
# if os.path.exists("save_model"):
#     saver.restore(sess,"save_model/model")

def optimistic_restore(session, save_file):
  """
  restore only those variable that exists in the model
  :param session:
  :param save_file:
  :return:
  """
  reader = tf.train.NewCheckpointReader(save_file)
  # reader.get_tensor()
  saved_shapes = reader.get_variable_to_shape_map()
  print(saved_shapes)
  print()
  print([var.name for var in tf.global_variables()])

  restore_vars = { (v.name.split(':')[0].replace("Model/","").replace("Model/","")): v for v in tf.trainable_variables() if 'bert' in v.name}
  saver = tf.train.Saver(restore_vars)
  saver.restore(session, save_file)

optimistic_restore(sess, "../uncased_L-12_H-768_A-12/bert_model.ckpt")

correct_count = 0
total_count = 0
max_acc = 0
best_accuracy = -1
for epoch in range(50):
    print('Train in epoch %d' % epoch)
    num_batch = len(train_data_util.all_batch)
    start_time = time.time()
    total_loss = 0
    for batch_index in range(num_batch):  # for each batch
        cur_batch = train_data_util.next_batch()
        train_feed_dict = {
            train_model.input_placeholder:cur_batch.input_batch,
            train_model.target2_placeholder:cur_batch.target_batch2,
            train_model.target1_placeholder:cur_batch.target_batch1,
            train_model.type_placeholder:cur_batch.type_batch,
            train_model.input_mask_placeholder:cur_batch.input_mask_batch,
            train_model.context_len_placeholder:cur_batch.context_len_batch,
            train_model.question_len_placeholder:cur_batch.question_len_batch,
            train_model.header_len_placeholder:cur_batch.header_len_batch,
        }
        _, loss_value,start_predict_logits,end_predict_logits = sess.run([train_model.train_op,
                                                          train_model.loss,
                                                          train_model.output_logits1,
                                                          train_model.output_logits2],
                                 feed_dict=train_feed_dict)
        total_loss += loss_value
        for i in range(len(start_predict_logits)):
            predict_start_index = np.argmax(start_predict_logits[i])
            true_start_index = np.argmax(cur_batch.target_batch1[i])

            predict_end_index = np.argmax(end_predict_logits[i])
            true_end_index = np.argmax(cur_batch.target_batch2[i])

            if predict_start_index == true_start_index\
                    and predict_end_index == true_end_index \
                or predict_start_index < config["max_input_len"] and \
                    (cur_batch.input_batch[i][predict_start_index:predict_end_index+1]==
                     cur_batch.input_batch[i][true_start_index:true_end_index+1]):
                correct_count += 1
            total_count += 1
    if total_count!=0:
        print("train acc" , (correct_count / total_count))
    duration = time.time() - start_time
    start_time = time.time()
    print('train loss = %.4f (%.3f sec)' % (total_loss / num_batch, duration))

    correct_count = 0
    total_count = 0
    total_loss = 0

    num_valid_batch = len(valid_data_util.all_batch)
    for valid_batch_index in range(num_valid_batch):
        cur_valid_batch = valid_data_util.next_batch()
        valid_feed_dict = {
            valid_model.input_placeholder: cur_valid_batch.input_batch,
            valid_model.target2_placeholder: cur_valid_batch.target_batch2,
            valid_model.target1_placeholder: cur_valid_batch.target_batch1,
            valid_model.type_placeholder: cur_valid_batch.type_batch,
            valid_model.input_mask_placeholder: cur_valid_batch.input_mask_batch,
            valid_model.context_len_placeholder: cur_valid_batch.context_len_batch,
            valid_model.question_len_placeholder: cur_valid_batch.question_len_batch,
            valid_model.header_len_placeholder: cur_valid_batch.header_len_batch,
        }
        valid_loss_value, start_predict_logits, end_predict_logits = sess.run(
            [valid_model.loss, valid_model.output_logits1, valid_model.output_logits2],
            feed_dict=valid_feed_dict)
        total_loss += valid_loss_value
        for i in range(len(start_predict_logits)):
            predict_start_index = np.argmax(start_predict_logits[i])
            true_start_index = np.argmax(cur_valid_batch.target_batch1[i])

            predict_end_index = np.argmax(end_predict_logits[i])
            true_end_index = np.argmax(cur_valid_batch.target_batch2[i])

            if predict_start_index == true_start_index\
                    and predict_end_index == true_end_index \
                or predict_start_index < config["max_input_len"] and \
                    (cur_valid_batch.input_batch[i][predict_start_index:predict_end_index+1]==
                     cur_valid_batch.input_batch[i][true_start_index:true_end_index+1]):
                correct_count += 1
            total_count += 1
    duration = time.time() - start_time
    start_time = time.time()
    print('valid loss = %.4f (%.3f sec)' % (total_loss / num_valid_batch, duration))
    if total_count!=0:
        print("valid acc" , (correct_count / total_count))
    if total_count != 0:
        if correct_count / total_count > max_acc and epoch>5:
            max_acc = correct_count / total_count
            saver.save(sess,"save_model/model")
    correct_count = 0
    total_count = 0
print("max acc",max_acc)