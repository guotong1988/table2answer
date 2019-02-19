import json

import numpy as np

def make_batch(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)] # zgwang: starting point of each batch

def pad_2d(in_vals, dim1_size, dim2_size, dtype=np.int32):
    out_val = np.zeros((dim1_size, dim2_size), dtype=dtype)
    if dim1_size > len(in_vals): dim1_size = len(in_vals)
    for i in range(dim1_size):
        cur_in_vals = in_vals[i]
        cur_dim2_size = dim2_size
        if cur_dim2_size > len(cur_in_vals): cur_dim2_size = len(cur_in_vals)
        out_val[i,:cur_dim2_size] = cur_in_vals[:cur_dim2_size]
    return out_val

def pad_3d(in_vals, dim1_size, dim2_size, dim3_size, dtype=np.int32):
    out_val = np.zeros((dim1_size, dim2_size, dim3_size), dtype=dtype)
    if dim1_size > len(in_vals): dim1_size = len(in_vals)
    for i in range(dim1_size):
        in_vals_i = in_vals[i]
        cur_dim2_size = dim2_size
        if cur_dim2_size > len(in_vals_i): cur_dim2_size = len(in_vals_i)
        for j in range(cur_dim2_size):
            in_vals_ij = in_vals_i[j]
            cur_dim3_size = dim3_size
            if cur_dim3_size > len(in_vals_ij): cur_dim3_size = len(in_vals_ij)
            out_val[i, j, :cur_dim3_size] = in_vals_ij[:cur_dim3_size]
    return out_val


class OneBatch(object):
    def __init__(self, current_batch, config):
        self.target_batch1 = []
        self.target_batch2 = []
        self.input_batch = []
        self.type_batch = []
        self.header_len_batch = []
        self.context_len_batch = []
        self.question_len_batch = []
        self.input_mask_batch = []
        # self.answer_batch = []
        for (answer_start,answer_end,input_id,type_id,
            header_len,context_len,question_len,input_mask) in current_batch:
            self.input_batch.append(input_id)
            self.target_batch1.append(answer_start)
            self.target_batch2.append(answer_end)
            self.type_batch.append(type_id)
            self.header_len_batch.append(header_len)
            self.context_len_batch.append(context_len)
            self.question_len_batch.append(question_len)
            self.input_mask_batch.append(input_mask)
        """
        to numpy
        """
        # self.all_cell_batch=np.array(self.all_cell_batch, dtype=np.int32)
        # self.question_batch = np.array(self.question_batch, dtype=np.int32)
        # self.target_batch = np.array(self.target_batch, dtype=np.int32)
        """
        padding
        """
        # self.all_cell_batch=pad_3d(self.all_cell_batch,dim1_size=config["batch_size"],
        #                            dim2_size=config["max_cell_num"],dim3_size=config["max_word_num"])
        # self.question_batch=pad_2d(self.question_batch,dim1_size=config["batch_size"],
        #                            dim2_size=config["max_question_len"])
        # self.target_batch = pad_2d(self.target_batch, dim1_size=config["batch_size"],
        #                            dim2_size=config["max_cell_num"]+config["max_question_len"])
        # self.type_batch = np.array(self.type_batch)


class DataUtil_bert(object):
    def __init__(self, json_path=None,config=None,tokenizer=None):

        f = open(json_path, mode="r", encoding="utf-8")
        jdata = json.load(f)
        all_case_list = []
        count = 0
        for context_qas in jdata["data"][0]["paragraphs"]:
            count += 1
            if config["debug"]==True and count>100:
                break
            if count % 1000==0:
                print("data process ", count)
            context = []
            for row in context_qas["context"]:
                context.extend(row)
            context_id = []
            context_id.append(tokenizer.convert_tokens_to_ids(["[CLS]"]))
            for cell in context:
                context_id.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(cell)))
            context_id.append(tokenizer.convert_tokens_to_ids(["[SEP]"]))

            for qas in context_qas["qas"]:
                question = tokenizer.tokenize(qas["question"])
                question = question + ["[SEP]"]
                question_id = tokenizer.convert_tokens_to_ids(question)
                # answer_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(qas["answers"][0]["text"]))

                flat_context_id = []
                word_index = 0
                answer_start = [0] * config["max_input_len"]
                answer_end = [0] * config["max_input_len"]

                for cell_index,cell in enumerate(context_id):
                    if word_index > 200:
                       break
                    if cell_index == qas["answers"][0]["answer_start"] + 1:
                        answer_start[word_index]=1
                    if cell_index == qas["answers"][0]["answer_start"] + 2:
                        answer_end[word_index-1]=1 # include or not
                    for word_id in cell:
                        flat_context_id.append(word_id)
                        word_index += 1

                header_len = len(context_qas["context"][0]) + 1
                context_len = len(flat_context_id)
                question_len = len(question_id)

                input_mask = (context_len + question_len)*[1] + (config["max_input_len"]-context_len-question_len)*[0]

                type_id = context_len*[0] + question_len*[1] + (config["max_input_len"]-context_len-question_len)*[0]

                input_id = flat_context_id + question_id + (config["max_input_len"]-context_len-question_len)*[0]

                if len(input_id) > config["max_input_len"]:
                    continue

                all_case_list.append((answer_start,answer_end,input_id,type_id,
                                      header_len,context_len,question_len,input_mask))
        print("data num", len(all_case_list))
        batch_spans = make_batch(len(all_case_list), config["batch_size"] )
        self.all_batch = []
        for batch_index, (batch_start, batch_end) in enumerate(batch_spans):
            current_batch = []
            for i in range(batch_start, batch_end):
                current_batch.append(all_case_list[i])
            if len(current_batch)<config["batch_size"]:
                 continue
            self.all_batch.append(OneBatch(current_batch,config))

        self.index_array = np.arange(len(self.all_batch))
        self.cur_pointer = 0

    def next_batch(self):
        if self.cur_pointer >= len(self.all_batch):
            self.cur_pointer = 0
            np.random.shuffle(self.index_array)
        cur_batch = self.all_batch[self.index_array[self.cur_pointer]]
        self.cur_pointer += 1
        return cur_batch

    def get_batch(self, i):
        if i >= len(self.all_batch): return None
        return self.all_batch[self.index_array[i]]