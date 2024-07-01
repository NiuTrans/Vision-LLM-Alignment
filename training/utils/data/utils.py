import torch
from torch.utils.data import Subset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import shutil
from torch.utils.data.dataloader import default_collate
import utils.data.DST as DST

NUM_DEBUG_SAMPLE = 10

def split_dataset(dataset, split_ratio=0.8):
    split = int(len(dataset) * split_ratio)
    return Subset(dataset, range(split)), Subset(dataset, range(split, len(dataset)))

def shuffle_dataset(dataset, np_rng):
    size = len(dataset)
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx)
    return Subset(dataset, shuffle_idx.tolist())

def save_debug_image(image_path, data_debug_path, data_debug_counter, rank, img_idx=0, base64=False):
    if data_debug_path is not None and data_debug_counter < NUM_DEBUG_SAMPLE:
        if base64:
            with open(f"{data_debug_path}/gpu_rank{rank}_debug{data_debug_counter}_image{img_idx}.jpg", 'wb') as f:
                f.write(image_path)
        else:
            shutil.copyfile(
                image_path,
                f"{data_debug_path}/gpu_rank{rank}_debug{data_debug_counter}_image{img_idx}.jpg")

def save_debug_text(text_to_save, data_debug_path, data_debug_counter, rank):
    if data_debug_path is not None and data_debug_counter < NUM_DEBUG_SAMPLE:
        with open(f"{data_debug_path}/gpu_rank{rank}_debug{data_debug_counter}_text.txt", 'w') as f:
            f.write(f"{text_to_save}")

class DataCollatorPadToMaxLen:

    def __init__(self, max_token_len, pad_token_id):
        self.max_token_len = max_token_len
        self.pad_token_id = pad_token_id

    def __call__(self, data):
        batch = {}
        input_ids = pad_sequence([default_collate(f['input_ids']) for f in data], 
                                  padding_value=self.pad_token_id, 
                                  batch_first=True)
        
        labels = pad_sequence([default_collate(f['labels']) for f in data],
                                   padding_value=DST.DEFAULT_LABEL_PADDING_NUM,
                                   batch_first=True)
        attention_mask = pad_sequence([default_collate(f['attention_mask']) for f in data],
                                        padding_value=0,
                                        batch_first=True)
        image = torch.concat([default_collate(f['image']) for f in data], dim=0).reshape((-1,) + data[0]["image"][0].shape[-3:])
        image_num = [f['image_num'] for f in data] 
        batch['input_ids'] = input_ids
        batch['labels'] = labels
        batch['attention_mask'] = attention_mask
        batch['image'] = image
        batch['image_num'] = image_num
        return batch

class DataCollatorPadToMaxLenForRewardModel:

    def __init__(self, max_token_len, pad_token_id):
        self.max_token_len = max_token_len
        self.pad_token_id = pad_token_id

    def __call__(self, data):
        batch = {}
        # data = data[0]
        data = [data[i][j] for i in range(len(data)) for j in range(len(data[0]))]

        input_ids = pad_sequence([default_collate(f['input_ids']) for f in data], 
                                  padding_value=self.pad_token_id, 
                                  batch_first=True)
        
        labels = pad_sequence([default_collate(f['labels']) for f in data],
                                   padding_value=DST.DEFAULT_LABEL_PADDING_NUM,
                                   batch_first=True)
        attention_mask = pad_sequence([default_collate(f['attention_mask']) for f in data],
                                        padding_value=0,
                                        batch_first=True)

        image = torch.concat([default_collate(f['image']) for f in data], dim=0).reshape((-1,) + data[0]["image"][0].shape[-3:])
        image_num = [f['image_num'] for f in data] 
        batch['input_ids'] = input_ids
        batch['labels'] = labels
        batch['attention_mask'] = attention_mask
        batch['image'] = image
        batch['image_num'] = image_num
        return batch

class DataCollatorPadToMaxLenForPPOTraining:

    def __init__(self, max_token_len, pad_token_id):
        self.max_token_len = max_token_len
        self.pad_token_id = pad_token_id

    def __call__(self, data):
        batch = {}

        input_ids = pad_sequence([default_collate(torch.flip(torch.LongTensor(f['input_ids']), dims=[-1]).tolist()) for f in data],
                                padding_value=self.pad_token_id,
                                batch_first=True)
        input_ids = torch.flip(input_ids, dims=[-1])
        
        labels = pad_sequence([default_collate(torch.flip(torch.LongTensor(f['labels']), dims=[-1]).tolist()) for f in data],
                                    padding_value=DST.DEFAULT_LABEL_PADDING_NUM,
                                    batch_first=True)
        labels = torch.flip(labels,  dims=[-1])

        attention_mask = pad_sequence([default_collate(torch.flip(torch.LongTensor(f['attention_mask']), dims=[-1]).tolist()) for f in data],
                                        padding_value=0,
                                        batch_first=True)
        attention_mask = torch.flip(attention_mask,  dims=[-1])

        image = torch.concat([default_collate(f['image']) for f in data], dim=0).reshape((-1,) + data[0]["image"][0].shape[-3:])
        image_num = [f['image_num'] for f in data]
        batch['input_ids'] = input_ids
        batch['labels'] = labels
        batch['attention_mask'] = attention_mask
        batch['image'] = image
        batch['image_num'] = image_num
        return batch

class DataCollatorPadToMaxLenForPrediction:

    def __init__(self, max_token_len, pad_token_id):
        self.max_token_len = max_token_len
        self.pad_token_id = pad_token_id

    def __call__(self, data):
        batch = {}

        input_ids = data[0]['input_ids']
        labels = data[0]['labels']
        attention_mask = data[0]['attention_mask']
        sample_id = data[0]['id']

        if len(data[0]['image']) == 0:
            image = torch.ones(1, 3, 244, 244) # a fake image
            image_num = [0]
        else:
            image = torch.concat([default_collate(f['image']) for f in data], dim=0).reshape((-1,) + data[0]["image"][0].shape[-3:])
            image_num = [data[0]['image_num']]

        batch['input_ids'] = torch.LongTensor(input_ids).unsqueeze(0)
        batch['labels'] = torch.LongTensor(labels).unsqueeze(0)
        batch['attention_mask'] = torch.LongTensor(attention_mask).unsqueeze(0)
        batch['image'] = image
        batch['image_num'] = image_num
        batch['id'] = sample_id
        return batch