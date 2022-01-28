import torch
from torch.utils.data import Dataset
from bert_data_utils import split_phone_posts

FRAME_SEQ_PED_TO = 1024
FRAME_DIM = 939
PAD = 0
FRAME_LIMIT = 1022

class PhoneTestDataset(Dataset):

    def __init__(self, mode: str, test_folder: str = "test"):

        assert(mode == 'test')
        self.mode = mode
        print(f"Loading {mode} phone posts from {test_folder}")
        self.phone_posts = split_phone_posts(
                path=test_folder, frame_limit=FRAME_LIMIT, position_dependent=True, show_message=True)
        self.ids = list(self.phone_posts.keys())
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx]

    def pad_sequence(self, sequence, pad_size, pad_seq):
        if len(sequence) < pad_size:
            sequence = sequence + pad_seq * (pad_size - len(sequence))
        return sequence

    def add_speical_token_dim(self, sequence):
        # Add 3 dim for CLS, SEP, PAD to every sequence
        return [[0, 0, 0] + feature for feature in sequence]

    def add_cls_sep_to_dim(self, sequence, dim: int):
        assert(dim > 0)
        _cls = [1, 0, 0] + [0] * (dim - 3)
        _sep = [0, 1, 0] + [0] * (dim - 3)
        return [_cls] + sequence + [_sep]

    def generate_key_padding_mask(self, mask_size: int, total_size: int):
        # ignore padding ,padding -> 1, not padding -> 0
        return [0] * mask_size + [1] * (total_size - mask_size)

    def collate_fn(self, batch):
        phone_posts = []
        phone_post_ids = []
        phone_post_key_padding_masks = []
        for phone_post_id in batch:
            phone_post = self.phone_posts[phone_post_id]
            phone_post_ids.append(phone_post_id)
            # create mask
            # phone_post need to add 2 for cls and sep token
            phone_post_mask = self.generate_key_padding_mask(
                len(phone_post) + 2, FRAME_SEQ_PED_TO)

            # expend 3 dim for special token to every phone post
            phone_post = self.add_speical_token_dim(phone_post)

            # add cls and sep
            phone_post = self.add_cls_sep_to_dim(phone_post, dim=FRAME_DIM)

            # pad frame squence to 1024
            phone_post = self.pad_sequence(phone_post, pad_size=FRAME_SEQ_PED_TO, pad_seq=[
                                           [0, 0, 1] + [0] * (FRAME_DIM - 3)])

            # append to batch data
            phone_posts.append(phone_post)
            phone_post_key_padding_masks.append(phone_post_mask)
        return (phone_post_ids, torch.tensor(phone_posts), torch.tensor(phone_post_key_padding_masks).bool())


