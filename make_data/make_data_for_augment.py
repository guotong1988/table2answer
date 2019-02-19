import json
import datetime
import argparse
import numpy as np
from utils import *



f = open("squad_data/train-v1.1.json", mode="r", encoding="utf-8")
squad_data = json.load(f)

parser = argparse.ArgumentParser()
parser.add_argument('--toy', action='store_true', default=False,
                    help='If set, use small data; used for fast debugging.')
parser.add_argument('--sd', type=str, default='saved_model_kg',
                    help='set model save directory.')
parser.add_argument('--db_content', type=int, default=0,
                    help='0: use knowledge graph type, 1: use db content to get type info')
parser.add_argument('--train_emb', action='store_true',
                    help='Use trained word embedding for SQLNet.')
args = parser.parse_args()

N_word=600
B_word=42
if args.toy:
    USE_SMALL=True
    GPU=True
    BATCH_SIZE=15
else:
    USE_SMALL=False
    GPU=True
    BATCH_SIZE=64
TEST_ENTRY=(True, True, True)  # (AGG, SEL, COND)

sql_data, table_data, val_sql_data, val_table_data, \
test_sql_data, test_table_data, \
TRAIN_DB, DEV_DB, TEST_DB = load_dataset(use_small=USE_SMALL)



def process(indata1,indata2,indata3,outfile):
    DB = indata3
    engine = DBEngine(DB)
    sql_data = indata1
    table_data = indata2
    # point to the position of matrix
    def input_table_output_matrix(input):
        result = []
        header = input["header_tok"]
        result_header = []
        for item in header:
            concat_str = ""
            for str_ in item:
                concat_str += str_
                concat_str += " "
            concat_str = concat_str.strip()
            result_header.append(concat_str.lower())
        result.append(result_header)
        for row in input["rows"]:
            result.append([str(item).lower() for item in row])
        return result

    tableid2qa={}
    paragraphs = []


    for item in squad_data["data"]:
        for context_qas in item["paragraphs"]:
            context = context_qas["context"]
            qas = context_qas["qas"]
    import sys
    import traceback
    count = 0
    for i in range(len(sql_data)):
        try:
            # if i>100:
            #     break
            count += 1
            # if count==16070:
            #    print()
            one_sql = sql_data[i]
            if one_sql["sql"]["agg"]!=0:
                continue
            if len(one_sql["sql"]["conds"])>1:
                continue
            # only consider one condition and no agg situation
            answer = engine.execute(one_sql["table_id"], one_sql["sql"]["sel"],one_sql["sql"]["agg"],one_sql["sql"]["conds"])
            question = one_sql["question"]

            # table_info = table_data[one_sql["table_id"]]

            if one_sql["table_id"] in tableid2qa:
                qa = {}
                qa["question"] = question.lower()
                qa["answer"] = [str(item).lower() for item in answer]
                tableid2qa[one_sql["table_id"]].append(qa)
            else:
                tableid2qa[one_sql["table_id"]] = []
                qa = {}
                qa["question"] = question.lower()
                qa["answer"] = [str(item).lower() for item in answer]
                tableid2qa[one_sql["table_id"]].append(qa)
        except:
            # exc_type, exc_value, exc_traceback = sys.exc_info()
            # traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
            print(count)
            continue

    print(count)
    def find_answer_position_in_matrix(matrix,text,row1,row2,col1,col2):
        result = 0
        # if row1==1 and row2==1 and col1==0 and col2==0:
        if True:
            for row_index,row in enumerate(matrix):
                for col_index,one_col in enumerate(row):
                    if one_col==text:
                        return result
                    else:
                        try:
                            if one_col == str(text).split(".")[0]:
                                return result
                        except:
                            continue
                    result += 1
            print("error")
            print(matrix)
            print(text)
            print()
        else:
            for row_index,row in enumerate(matrix):
                if row_index==row1 or row_index==row2:
                    for col_index,one_col in enumerate(row):
                        if col_index==col1 or col_index==col2:
                            if one_col==text:
                                return result
                            else:
                                try:
                                    if one_col == str(text).split(".")[0]:
                                        return result
                                except:
                                    continue
                            result += 1

        return None


    paragraphs = []

    def add_to_paragraphs_with_shuffle(row1=1,row2=1,col1=0,col2=0):
      for tableid in tableid2qa:
        context = input_table_output_matrix(table_data[tableid])

        if col1<len(context[0]) and col2<len(context[0]):
            context_ = np.array(context)
            tmp_col = context_[:,col1].copy()
            context_[:,col1] = context_[:,col2]
            context_[:,col2] = tmp_col
            context = context_.tolist()
        else:
            continue

        if row1<len(context) and row2<len(context):
            tmp_row = context[row1]
            context[row1] = context[row2]
            context[row2] = tmp_row
        else:
            continue

        context_qas ={}
        context_qas["context"] = context
        context_qas["qas"] = []

        for item in tableid2qa[tableid]:
            one_qas = {}
            one_qas["answers"] = []
            one_answer = {}
            if len(item["answer"])==1:# only consider one answer
                one_answer["text"] = item["answer"][0]
            else:
                continue
            one_answer["answer_start"] = find_answer_position_in_matrix(context,one_answer["text"],row1,row2,col1,col2)
            if one_answer["answer_start"]==None:
                continue
            one_qas["question"] = item["question"]
            one_qas["answers"].append(one_answer)
            context_qas["qas"].append(one_qas)
        if len(context_qas["qas"])==0: # only context, no answer
            continue
        else:
            paragraphs.append(context_qas)
      print("end one")

    """    
    add_to_paragraphs_with_shuffle(col1=0, col2=1)
    add_to_paragraphs_with_shuffle(col1=0, col2=2)
    add_to_paragraphs_with_shuffle(col1=0, col2=3)
    add_to_paragraphs_with_shuffle(col1=0, col2=4)
    add_to_paragraphs_with_shuffle(col1=0, col2=5)
    add_to_paragraphs_with_shuffle(col1=0, col2=6)
    add_to_paragraphs_with_shuffle(col1=0, col2=7)
    add_to_paragraphs_with_shuffle(col1=1, col2=2)
    add_to_paragraphs_with_shuffle(col1=1, col2=3)
    add_to_paragraphs_with_shuffle(col1=1, col2=4)
    add_to_paragraphs_with_shuffle(col1=1, col2=5)
    add_to_paragraphs_with_shuffle(col1=1, col2=6)
    add_to_paragraphs_with_shuffle(col1=1, col2=7)
    add_to_paragraphs_with_shuffle(col1=2, col2=3)
    add_to_paragraphs_with_shuffle(col1=2, col2=4)
    add_to_paragraphs_with_shuffle(col1=2, col2=5)
    add_to_paragraphs_with_shuffle(col1=2, col2=6)
    add_to_paragraphs_with_shuffle(col1=2, col2=7)
    add_to_paragraphs_with_shuffle(col1=3, col2=4)
    add_to_paragraphs_with_shuffle(col1=3, col2=5)
    add_to_paragraphs_with_shuffle(col1=3, col2=6)
    add_to_paragraphs_with_shuffle(col1=3, col2=7)
    add_to_paragraphs_with_shuffle(col1=4, col2=5)
    add_to_paragraphs_with_shuffle(col1=4, col2=6)
    add_to_paragraphs_with_shuffle(col1=4, col2=7)
    add_to_paragraphs_with_shuffle(col1=5, col2=6)
    add_to_paragraphs_with_shuffle(col1=5, col2=7)
    add_to_paragraphs_with_shuffle(col1=6, col2=7)
    """
    add_to_paragraphs_with_shuffle()

    add_to_paragraphs_with_shuffle(1, 2)
    add_to_paragraphs_with_shuffle(1, 3)
    add_to_paragraphs_with_shuffle(1, 4)
    add_to_paragraphs_with_shuffle(1, 5)
    add_to_paragraphs_with_shuffle(1, 6)
    add_to_paragraphs_with_shuffle(1, 7)
    add_to_paragraphs_with_shuffle(1, 8)
    add_to_paragraphs_with_shuffle(1, 9)
    add_to_paragraphs_with_shuffle(1, 10)
    add_to_paragraphs_with_shuffle(2, 3)
    add_to_paragraphs_with_shuffle(2, 4)
    add_to_paragraphs_with_shuffle(2, 5)
    add_to_paragraphs_with_shuffle(2, 6)
    add_to_paragraphs_with_shuffle(2, 7)
    add_to_paragraphs_with_shuffle(2, 8)
    add_to_paragraphs_with_shuffle(2, 9)
    add_to_paragraphs_with_shuffle(2, 10)
    add_to_paragraphs_with_shuffle(3, 4)
    add_to_paragraphs_with_shuffle(3, 5)
    add_to_paragraphs_with_shuffle(3, 6)
    add_to_paragraphs_with_shuffle(3, 7)
    add_to_paragraphs_with_shuffle(3, 8)
    add_to_paragraphs_with_shuffle(3, 9)
    add_to_paragraphs_with_shuffle(3, 10)
    add_to_paragraphs_with_shuffle(4, 5)
    add_to_paragraphs_with_shuffle(4, 6)
    add_to_paragraphs_with_shuffle(4, 7)
    add_to_paragraphs_with_shuffle(4, 8)
    add_to_paragraphs_with_shuffle(4, 9)
    add_to_paragraphs_with_shuffle(4, 10)
    add_to_paragraphs_with_shuffle(5, 6)
    add_to_paragraphs_with_shuffle(5, 7)
    add_to_paragraphs_with_shuffle(5, 8)
    add_to_paragraphs_with_shuffle(5, 9)
    add_to_paragraphs_with_shuffle(5, 10)
    # add_to_paragraphs_with_shuffle(6, 7)
    # add_to_paragraphs_with_shuffle(6, 8)
    # add_to_paragraphs_with_shuffle(6, 9)
    # add_to_paragraphs_with_shuffle(6, 10)
    # add_to_paragraphs_with_shuffle(7, 8)
    # add_to_paragraphs_with_shuffle(7, 9)
    # add_to_paragraphs_with_shuffle(7, 10)
    # add_to_paragraphs_with_shuffle(8, 9)
    # add_to_paragraphs_with_shuffle(8, 10)
    # add_to_paragraphs_with_shuffle(9, 10)

    squad_data["data"]=[]
    one_data = {}
    one_data["paragraphs"] = paragraphs
    squad_data["data"].append(one_data)

    f2 = open(outfile, mode="w", encoding="utf-8")
    json.dump(squad_data,f2)

process(sql_data,table_data,TRAIN_DB,"squad_data/my_train.json")