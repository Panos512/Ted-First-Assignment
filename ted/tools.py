from os import path
import pandas as pd

def open_csv(relative_path='./data_sets/train_set.csv'):
    d = path.dirname(__file__)
    train_set_path = path.join(d, relative_path)
    df = pd.read_csv(train_set_path, sep='\t')
    return df
