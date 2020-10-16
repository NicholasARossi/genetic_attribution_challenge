import pytest
from .DNA_CNN_utils import seq_prep,convert_DNA_onehot2D,is_nuc_acid
import numpy as np

@pytest.fixture
def single_seqs():
    """ returns the squence data dictionary """
    return {'DNA':'AGACTGTTGACGGCCT',
            'not_DNA':'BANANAS',
            'pad2_RC_DNA':'NNAGACTGTTGACGGCCTNNNNNNNNNNAGGCCGTCAACAGTCTNN',
            'pad0_RC_DNA_short':'AGACTAGTCT',
            'pad2_noRC_DNA_long':'NNAGACTGTTGACGGCCTNNNNNNNNNNNNNNNN'}

@pytest.fixture
def sequence_list():
    ''' Returns list of '''
    return {'input_list':['AGCT','AGCTA'],
            'output_encoding':np.array([[[1., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                                      [0., 1., 0., 0., 0., 0., 0., 1., 0., 0.],
                                      [0., 0., 1., 0., 0., 0., 0., 0., 1., 0.],
                                      [0., 0., 0., 1., 0., 0., 0., 0., 0., 1.]],

                                     [[1., 0., 0., 0., 1., 0., 1., 0., 0., 0.],
                                      [0., 1., 0., 0., 0., 0., 0., 1., 0., 0.],
                                      [0., 0., 1., 0., 0., 0., 0., 0., 1., 0.],
                                      [0., 0., 0., 1., 0., 1., 0., 0., 0., 1.]]]),
            'broken_list':['banana'],
            'broken_encoding':np.array([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]])}

def test_is_nuc_acid(single_seqs):
    ''' Test function for determining if something'''
    assert is_nuc_acid(single_seqs['DNA']) is not False
    assert is_nuc_acid(single_seqs['not_DNA']) is False


def test_seq_prep(single_seqs):
    """ test for RNA converter """
    # if the sequence is shorter than the maximum sequence
    assert single_seqs['pad2_RC_DNA']==seq_prep(single_seqs['DNA'],20,2)

    # if the sequence is longer than tht maximum sequence
    assert single_seqs['pad0_RC_DNA_short'] == seq_prep(single_seqs['DNA'], 5, 0)

    # if the seqence is shorter but no reverse complement
    assert single_seqs['pad2_noRC_DNA_long'] == seq_prep('AGACTGTTGACGGCCT',30,2,rev_comp='False')

def test_convert_DNA_onehot2D(sequence_list):
    encoded_DNA=convert_DNA_onehot2D(sequence_list['input_list'])
    assert np.array_equal(encoded_DNA,sequence_list['output_encoding'])

    non_DNA=convert_DNA_onehot2D(sequence_list['broken_list'])
    assert np.array_equal(non_DNA,sequence_list['broken_encoding'])




    #assert convert_RNA(sequences['RNA'])==sequences['DNA']