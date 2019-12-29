import os
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from torch.utils.data import Dataset
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold

    
def read_ZINC(num_mol):
    f = open('../Data/logP/ZINC.smiles', 'r')
    contents = f.readlines()

    list_smi = []
    fps = []
    logP = []
    tpsa = []
    for i in range(num_mol):
        smi = contents[i].strip()
        list_smi.append(smi)
        m = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(m,2)
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp,arr)
        fps.append(arr)
        logP.append(MolLogP(m))
        tpsa.append(CalcTPSA(m))

    fps = np.asarray(fps).astype(float)
    logP = np.asarray(logP).astype(float)
    tpsa = np.asarray(tpsa).astype(float)

    return list_smi, logP, tpsa


class myDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
def make_partition(args):
    smiles, logPs, _ = read_ZINC(args.num_mol)
    
    # Truncate smile strings
    for i, smile in enumerate(smiles):
        truncated_smile = smile[:args.max_len]
        filled_smile = truncated_smile + ' '* (args.max_len-len(truncated_smile))
        smiles[i] = filled_smile
        
    X, y = np.array(smiles), np.array(logPs)

    list_fold = list()
    cv = KFold(n_splits=args.n_splits, random_state=args.seed)
    for train_index, test_index in cv.split(X, y):
        # Spliting data into train & test set 
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Spliting train data into train & validation set
        X_train, X_val, y_train, y_val = train_test_split(X_train, 
                                                          y_train, 
                                                          test_size=args.test_size,
                                                          random_state=args.seed)

        # Construct dataset object
        train_set = myDataset(X_train, y_train)
        val_set = myDataset(X_val, y_val)
        test_set = myDataset(X_test, y_test)

        partition = {
            'train': train_set,
            'val': val_set,
            'test': test_set
        }
        list_fold.append(partition)
    return list_fold


from decimal import Decimal
import json 
from os import listdir
from os.path import isfile, join
import pandas as pd
import hashlib

class Writer():
    
    def __init__(self, prior_keyword=[], dir='./results'):
        self.prior_keyword = prior_keyword
        self.dir = dir
        
    def generate_hash(self, args):
        str_as_bytes = str.encode(str(args))
        hashed = hashlib.sha256(str_as_bytes).hexdigest()[:24]
        return hashed

    def write(self, args, prior_keyword=None):
        dict_args = vars(args)
        if 'bar' in dict_args:
            #del dict_args['bar']
            pass
        
        if prior_keyword:
            self.prior_keyword = prior_keyword
        filename = 'exp_{}'.format(args.exp_name)
        for keyword in self.prior_keyword:
            value = str(dict_args[keyword])
            if value.isdigit():
                filename += keyword + ':{:.2E}_'.format(Decimal(dict_args[keyword]))
            else:
                filename += keyword + ':{}_'.format(value)
#         hashcode = self.generate_hash(args)
#         filename += hashcode
        filename += '.json'
        
        with open(self.dir+'/'+filename, 'w') as outfile:
            json.dump(dict_args, outfile)
            
    def read(self, exp_name=''):
        list_result = list()
        filenames = [f for f in listdir(self.dir) if isfile(join(self.dir, f))]
        for filename in filenames:
            with open(join(self.dir, filename), 'r') as infile:
                result = json.load(infile)
                if len(exp_name) > 0:
                    if result['exp_name'] == exp_name:
                        list_result.append(result)
                else:
                    list_result.append(result)
                        
        return pd.DataFrame(list_result)
    
    def clear(self, exp_name=''):
        filenames = [f for f in listdir(self.dir) if isfile(join(self.dir, f))]
        for filename in filenames:
            if len(exp_name) > 0:
                result = json.load(open(join(self.dir, filename), 'r'))
                if result['exp_name'] == exp_name:
                    os.remove(join(self.dir, filename))
            else:
                os.remove(join(self.dir, filename))