from torch.utils.data import Dataset
import numpy as np


class Dataset_npy(Dataset):
    """ 
    Dataset loader for binary numpy files.
    """
    def __init__(self, filepath : str, kmer : int = 51, transform : object = None ) -> Dataset:
        self._data = np.load(filepath)
        self._kmer = kmer
        self._transform = transform
    def __getitem__(self, idx : int) -> tuple[np.array]:
        kmer = self._data[idx][:self._kmer]
        signal = np.reshape(self._data[idx][self._kmer : -1], (1, self._kmer, -1))
        label = self._data[idx][-1]
        return kmer, signal, label 
        
    def __len__(self,) -> int:
        return self._data.shape[0]