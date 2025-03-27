import numpy as np

## Set the code of English words

word_SOS = [1,0,0,0]
word_Let = [0,1,0,0]
word_to = [0,0,1,0]
word_go = [0,0,0,1]

word2vec = np.array(
    [[1.16, -0.77], 
     [-0.27,0.82],
     [-2.19, 0.89],
     [3.5, -1.74]]
)
word2vecT = np.transpose(word2vec)

sentence_matrix = np.array(
    [word_SOS,
     word_Let,
     word_go]
)


## Encoder

### Step 1: Word