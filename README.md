# table2answer

Table2answer: Read the database and answer without SQL 

https://arxiv.org/abs/1902.04260

# requirement

python3 

tensorflow >= 1.12.0

# Train

### Step 1. 

Download the pre-trained model at https://github.com/google-research/bert and unzip them to `uncased_L-12_H-768_A-12`

### Step 2.

Download the v1.1 squad data at https://github.com/rajpurkar/SQuAD-explorer/tree/master/dataset

### Step 3. 

use `make_data` to create the data.

### Step 4. 

`matrix_code/train.py`

