import pandas as pd
from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    def __init__(self, df: pd.DataFrame):
        pass


if __name__ == '__main__':
    df = pd.read_csv('../sample.csv')
    dataset = Dataset(df)
