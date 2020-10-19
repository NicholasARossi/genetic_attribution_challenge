import numpy as np
import pandas as pd
from DNA_CNN_utils import convert_DNA_onehot2D
from tqdm import tqdm
import gzip
import os


from sklearn.utils import class_weight,shuffle
from sklearn.model_selection import train_test_split

def prep_thicc_data(in_dir='../data/',out_dir='../data/'):

    if not os.path.exists(f'{out_dir}training_data'):
        os.makedirs(f'{out_dir}training_data')

    if not os.path.exists(f'{out_dir}validation_data'):
        os.makedirs(f'{out_dir}validation_data')

    if not os.path.exists(f'{out_dir}testing_data'):
        os.makedirs(f'{out_dir}testing_data')


    ## loading the data
    train_values=pd.read_csv(f'{in_dir}train_values.csv').head(101)
    test_values = pd.read_csv(f'{in_dir}test_values.csv').head(101)
    train_labels = pd.read_csv(f'{in_dir}train_labels.csv').head(101)
    y = train_labels[train_labels.columns[1:]]
    sequence_id=test_values.sequence_id




    ##compute class weight
    class_weights = class_weight.compute_class_weight('balanced',
                                                     np.unique(y.idxmax(1)),
                                                     y.idxmax(1))

    np.save(f'{out_dir}class_weights.npy',class_weights)



    train_dna_seqs=train_values.sequence.values
    test_dna_seqs=test_values.sequence.values
    ### truncated extra long:
    cutoff=10000
    train_dna_seqs=[x[:cutoff] for x in train_dna_seqs]
    test_dna_seqs=[x[:cutoff] for x in train_dna_seqs]


    y=np.asarray(y)

    ## computing the maximum length of any of the DNA sequences
    #max_len=max([len(x) for x in np.append(test_dna_seqs,train_dna_seqs)])
    max_len=cutoff
    padding=48

    # save validation set
    print('saving validation data... \n')
    train_dna_seqs, train_dna_seqs_validation, y, y_validation = train_test_split(train_dna_seqs, y, test_size=0.001, random_state=88)
    f = gzip.GzipFile(f"{out_dir}validation_data/values.npy.gz", "w")
    np.save(file=f,arr=convert_DNA_onehot2D(train_dna_seqs_validation,max_len=max_len,padding=padding)[..., np.newaxis])
    np.save(f"{out_dir}validation_data/labels.npy", y_validation)




    num_training_seqs = len(train_dna_seqs)
    num_chunks = 10
    chunk_size = int(num_training_seqs / num_chunks)

    # compute the maxium length for all the samples

    print('Spliting ands saving Training Data ... \n')
    for z in tqdm(range(num_chunks)):
        f = gzip.GzipFile(f"../data/training_data/values_{z}.npy.gz", "w")

        np.save(file=f,arr=convert_DNA_onehot2D(train_dna_seqs[z * chunk_size:(z + 1) * chunk_size],max_len=max_len,padding=padding)[..., np.newaxis])
        np.save(f"{out_dir}training_data/labels_{z}.npy",y[z*chunk_size:(z+1)*chunk_size])

    try:
        z=num_chunks
        f = gzip.GzipFile(f"../data/training_data/values_{z}.npy.gz", "w")

        np.save(file=f, arr=
        convert_DNA_onehot2D(train_dna_seqs[z * chunk_size:], max_len=max_len, padding=padding)[
            ..., np.newaxis])
        np.save(f"{out_dir}training_data/labels_{z}.npy", y[z * chunk_size:])
    except:
        pass

    num_testing_seqs = len(test_dna_seqs)
    chunk_size = int(num_testing_seqs / num_chunks)

    print('Spliting and saving Testing Data ... \n')

    for z in tqdm(range(num_chunks)):
        f = gzip.GzipFile(f"{out_dir}testing_data/values_{z}.npy.gz", "w")

        np.save(file=f,arr=convert_DNA_onehot2D(test_dna_seqs[z * chunk_size:(z + 1) * chunk_size],max_len=max_len,padding=padding)[..., np.newaxis])
        np.save(f"{out_dir}testing_data/labels_{z}.npy",sequence_id[z*chunk_size:(z+1)*chunk_size])

    try:
        z=num_chunks
        f = gzip.GzipFile(f"{out_dir}testing_data/values_{z}.npy.gz", "w")

        np.save(file=f,arr=convert_DNA_onehot2D(test_dna_seqs[z * chunk_size:],max_len=max_len,padding=padding)[..., np.newaxis])


        np.save(f"{out_dir}testing_data/labels_{z}.npy", sequence_id[z * chunk_size:])
    except:
        pass

if __name__ == '__main__':
    prep_thicc_data(out_dir='../big_chunks/')