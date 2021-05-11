# table2answer

[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)

Table2answer: Read the database and answer without SQL 

https://arxiv.org/abs/1902.04260

# REASONABLE

The reason why we think removing the logic form step is possible is that human can do the text2sql task without explicit logic form.

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

