from genericpath import exists
import gdown
import tarfile
import os

def download_dataset():
    url = 'https://drive.google.com/uc?id=1AdMbVK110IKLG7wJKhga2N2fitV1bVPA'
    output = 'drinks.tar.gz'
    gdown.download(url, output, quiet=False)
    file = tarfile.open('drinks.tar.gz')
    file.extractall('./')
    file.close()
    print("Finished downloading drinks dataset")