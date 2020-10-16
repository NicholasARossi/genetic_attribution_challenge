import numpy as np
import pandas as pd

from DNA_CNN_utils import convert_DNA_onehot2D
from tqdm import tqdm
import gzip


upper_dir = '../data/'
train_values=pd.read_csv(f'{upper_dir}train_values.csv')
test_values = pd.read_csv(f'{upper_dir}test_values.csv')
train_labels = pd.read_csv(f'{upper_dir}train_labels.csv')

y = train_labels[train_labels.columns[1:]]
y=np.asarray(y)
train_dna_seqs=train_values.sequence.values
test_dna_seqs=test_values.sequence.values

num_training_seqs = len(train_dna_seqs)
num_chunks = 100
chunk_size = int(num_training_seqs / num_chunks)

# compute the maxium length for all the samples
max_len=max([len(x) for x in np.append(test_dna_seqs,train_dna_seqs)])

print('Spliting Training Data ... \n')
for z in tqdm(range(num_chunks)):
    f = gzip.GzipFile(f"../data/training_data/values_{z}", "w")

    np.save(file=f,arr=convert_DNA_onehot2D(train_dna_seqs[z * chunk_size:(z + 1) * chunk_size],max_len=max_len)[..., np.newaxis])
    np.save(f"../data/training_data/labels_{z}",y[z*chunk_size:(z+1)*chunk_size])


num_testing_seqs = len(test_dna_seqs)
num_chunks = 20
chunk_size = int(num_testing_seqs / num_chunks)

print('Spliting Testing Data ... \n')

for z in tqdm(range(num_chunks)):
    f = gzip.GzipFile(f"../data/testing_data/values_{z}", "w")

    np.save(file=f,arr=convert_DNA_onehot2D(test_dna_seqs[z * chunk_size:(z + 1) * chunk_size],max_len=max_len)[..., np.newaxis])
