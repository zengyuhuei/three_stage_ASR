from bert_data_utils import load_ark
from glob import glob
from tqdm import tqdm
import logging
import os
import pickle
import argparse

loglevel = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                        level=loglevel, datefmt="%Y-%m-%d %H:%M:%S")

def load_and_save(_type:str, phone_post_path:str):
    paths = glob(f"{phone_post_path}/{_type}/phone_post.*.ark")
    
    # create folder to store preprocessed files
    if not os.path.isdir(f"../data/{_type}"):
        os.mkdir(f"../data/{_type}")
    if not os.path.isdir(f"../data/{_type}/phone_post"):
        os.mkdir(f"../data/{_type}/phone_post")
    
    logging.info(f"Found phone_post file nums: {len(paths)}")
    for i, path in enumerate(paths):
        logging.info(f"Progess {i+1}/{len(paths)}: {path}")
        phone_posts = load_ark(path, ppg_dim=935)
        for _id in tqdm(phone_posts):
            with open(f'../data/{_type}/phone_post/{_id}.pkl', 'wb') as f:
                pickle.dump(phone_posts[_id], f, protocol=pickle.HIGHEST_PROTOCOL)
        del phone_posts
def process_command():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ppg_path', dest="ppg_path")
    return parser.parse_args()

if __name__ == "__main__":
    args = process_command()
    for _type in ['matbn_dev_hires']:
        logging.info(f"Process : {_type}")
        # args.ppg_path: 存放phone_post.*.ark文件的路径
        load_and_save(_type, args.ppg_path)