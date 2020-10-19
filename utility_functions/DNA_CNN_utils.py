
import numpy as np


DNA_hot_code = {'A': [1, 0, 0, 0], 'G': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'T': [0, 0, 0, 1],
            'N': [0, 0, 0, 0]}


rc_dict = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}

VALID_NUC_ACID = ['A', 'C', 'G', 'T','N']

def is_nuc_acid(input_seq):
    try:
        return all(nuc in VALID_NUC_ACID for nuc in input_seq.upper())
    except Exception:
        return False

def seq_prep(seq,maxlen,padding,rev_comp=True):
    '''
    Helper Function that restructures a DNA sequence to be learnable, by default
    :param seq: input DNA sequence
    :param max_len: maximum length of all DNA sequences
    :param padding: optional padding to facilitate edge work with
    :param rev_comp:
    :return:
    '''

    # test to see if it's a nucleic acid - if not set it to empty
    seq=seq.upper()
    if not is_nuc_acid(seq):
        print(f'\n{seq} is not a nucleic acid!')
        seq=''




    if len(seq) > maxlen:
        seq = seq[0:maxlen]
    else:
        seq = seq

    fwd_seq = "N" *padding+ seq + "N" * (maxlen - len(seq))
    if rev_comp==True:
        complement_seq = ''
        for n in fwd_seq:
            complement_seq += rc_dict[n]

        complete_seq = fwd_seq + 'N' * padding + complement_seq[::-1]
        return complete_seq

    else:
        complete_seq = fwd_seq + 'N' * padding
        return complete_seq


def convert_DNA_onehot2D(list_of_seqs,DNA=True,rev_comp=True,padding=0,max_len=None):


    # start by converting all the seuences to the same length
    if max_len==None:
        max_len = max([len(seq) for seq in list_of_seqs])

    list_of_seqs=[seq_prep(seq,max_len,padding,rev_comp=rev_comp) for seq in list_of_seqs]


    list_of_onehot2D_seqs = np.zeros((len(list_of_seqs),4,len(list_of_seqs[0])))

    if DNA==True:
        nt_dict = DNA_hot_code

    count = 0
    for seq in list_of_seqs:
        if len(seq) > 1:
            for letter in range(len(seq)):
                for i in range(4):
                    list_of_onehot2D_seqs[count][i][letter] = (nt_dict[seq[letter]])[i]
        count += 1
    return list_of_onehot2D_seqs

if __name__ == '__main__':

    encoded_DNA=convert_DNA_onehot2D(['AGCT','AGCTA','banana'])
    print(encoded_DNA)
