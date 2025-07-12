import pandas as pd
import os
import zipfile
import urllib.request

def download_and_extract_movielens(data_dir='data/ml-100k'):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        zip_url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
        zip_path = os.path.join(data_dir, 'ml-100k.zip')
        urllib.request.urlretrieve(zip_url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('data/')

def load_movielens_100k(data_dir='data/ml-100k/'):
    download_and_extract_movielens()
    ratings_path = os.path.join(data_dir, 'u.data')
    items_path = os.path.join(data_dir, 'u.item')

    # load rating data
    ratings = pd.read_csv(ratings_path,sep='\t',names=['user_id', 'item_id', 'rating', 'timestamp'])

      # Optional: Load movie titles
    items = pd.read_csv(items_path, sep='|', encoding='latin-1', header=None, usecols=[0, 1], names=['item_id', 'title'])

    ratings=ratings.merge(items,on='item_id')

    return ratings


if __name__ == '__main__':
    df = load_movielens_100k()
    print(df.head())