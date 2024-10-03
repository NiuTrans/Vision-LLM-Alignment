import torch
from torch.utils.data import Subset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import shutil
from torch.utils.data.dataloader import default_collate
import training.utils.data.DST as DST

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

class DataCollatorPadToMaxLenForMSERewardModel:

    def __init__(self, max_token_len, pad_token_id, image_size):
        self.max_token_len = max_token_len
        self.pad_token_id = pad_token_id
        self.image_size = image_size

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
        
        image_num = []
        image_data = []
        score_list = []
        for single_data in data:
            if single_data['image'][0] is None:
                image_data.append(torch.zeros(1, 3, self.image_size['height'],self.image_size['width'])) 
                image_num.append(0)
            else:
                image_data.append(default_collate(single_data['image']))
                image_num.append(single_data['image_num'])
            score_list.append(single_data["score"])

        image = torch.concat(image_data, dim=0).reshape((-1,) + image_data[0].shape[-3:])
    
        batch['input_ids'] = input_ids
        batch['labels'] = labels
        batch['attention_mask'] = attention_mask
        batch['image'] = image
        batch['image_num'] = image_num
        batch['score'] = score_list
        return batch

class DataCollatorPadToMaxLen:

    def __init__(self, max_token_len, pad_token_id, image_size):
        self.max_token_len = max_token_len
        self.pad_token_id = pad_token_id
        self.image_size = image_size # {'height': 336, 'width': 336}

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
        image_num = []
        image_data = []
        image_sizes = []
        aspect_ratio_ids = []
        aspect_ratio_mask = []

        # check for the presence of multiple images?
        multi_tag = False
        max_image_number = 1

        for single_data in data:
            if single_data['image'] is None:
                if 'aspect_ratio_ids' in single_data.keys():
                    image_data.append(torch.zeros(1, 1, 4, 3, self.image_size['height'],self.image_size['width']))
                    aspect_ratio_ids.append(torch.LongTensor([[4]]))
                    aspect_ratio_mask.append(torch.LongTensor([[[1,0,0,0]]]))
                    image_num.append(1)
                else:
                    image_data.append(torch.zeros(1, 3, self.image_size['height'],self.image_size['width'])) 
                    image_num.append(1)
            else:
                if 'aspect_ratio_ids' in single_data.keys():
                    image_data.append(torch.Tensor(default_collate(single_data['image'])))    # {'image': [image_array]]}

                    try:
                        aspect_ratio_ids.append(torch.LongTensor(default_collate(single_data['aspect_ratio_ids'])))
                        aspect_ratio_mask.append(torch.LongTensor(default_collate(single_data['aspect_ratio_mask'])))
                    except:
                        aspect_ratio_ids.append(default_collate(single_data['aspect_ratio_ids']))
                        aspect_ratio_mask.append(default_collate(single_data['aspect_ratio_mask']))
                    
                    image_num.append(single_data['image_num'])
                    # check multiple images?
                    
                    if image_data[-1].size(1) > max_image_number:
                        max_image_number = image_data[-1].size(1)
                        multi_tag = True
                else:
                    image_data.append(default_collate(single_data['image'][0]))
                    image_num.append(single_data['image_num'])
        
        if multi_tag:  #for llama-3.2-vision
            for index in range(len(image_data)):
                if image_data[index].size(1) != max_image_number:
                    new_image = []
                    new_aspect_ratio_ids = []
                    new_aspect_ratio_mask = []

                    new_image.append(image_data[index])
                    new_aspect_ratio_ids.append(aspect_ratio_ids[index])
                    new_aspect_ratio_mask.append(aspect_ratio_mask[index])

                    for i in range(max_image_number-image_data[index].size(1)):
                        new_image.append(torch.zeros(1, 1, 4, 3, self.image_size['height'],self.image_size['width']))
                        new_aspect_ratio_ids.append(torch.LongTensor([[[4]]]))
                        new_aspect_ratio_mask.append(torch.LongTensor([[[[1,0,0,0]]]]))

                    new_image = torch.cat(new_image, dim=1)
                    new_aspect_ratio_ids = torch.cat(new_aspect_ratio_ids, dim=2)
                    new_aspect_ratio_mask = torch.cat(new_aspect_ratio_mask, dim=2)

                    image_data[index] =  new_image
                    aspect_ratio_ids[index] = new_aspect_ratio_ids
                    aspect_ratio_mask[index] = new_aspect_ratio_mask
        
        if 'aspect_ratio_ids' in data[0].keys():
            aspect_ratio_ids = torch.cat(aspect_ratio_ids, dim=1).squeeze(0)
            aspect_ratio_mask = torch.cat(aspect_ratio_mask, dim=1).squeeze(0)
            
            batch['aspect_ratio_ids'] = aspect_ratio_ids
            batch['aspect_ratio_mask'] = aspect_ratio_mask

        image = torch.cat(image_data, dim=0)

        batch['input_ids'] = input_ids
        batch['labels'] = labels
        batch['attention_mask'] = attention_mask
        batch['image'] = image
        batch['image_num'] = image_num
        return batch

class DataCollatorPadToMaxLenForRewardModel:

    def __init__(self, max_token_len, pad_token_id, image_size):
        self.max_token_len = max_token_len
        self.pad_token_id = pad_token_id
        self.image_size = image_size

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
        
        image_num = []
        image_data = []
        query_ids = []
        image_sizes = []
        aspect_ratio_ids = []
        aspect_ratio_mask = []
        for single_data in data:
            if single_data['image'][0] is None:
                if 'image_sizes' in single_data.keys():
                    image_data.append(torch.zeros(1, 5, 3, self.image_size['height'],self.image_size['width']))
                    image_sizes.append(torch.LongTensor([123, 123]))
                elif 'aspect_ratio_ids' in data[0].keys():
                    image_data.append(torch.zeros(1, 4, 3, self.image_size['height'],self.image_size['width']))
                    aspect_ratio_ids.append(torch.LongTensor([4]))
                    aspect_ratio_mask.append(torch.LongTensor([[0,0,0,0]]))
                else:
                    image_data.append(torch.zeros(1, 3, self.image_size['height'],self.image_size['width']))
                image_num.append(0)
            else:
                if 'image_sizes' in single_data.keys():
                    if len(single_data['image_sizes']) != 0:
                        image_sizes.append(default_collate(single_data['image_sizes']))
                        if default_collate(single_data['image']).size(1) == 5:
                            image_data.append(default_collate(single_data['image']))
                        else:
                            tmp_image = default_collate(single_data['image'])
                            current_dim_size = tmp_image.size(1)
                            padding_size = 5 - current_dim_size
                            tmp_image = torch.nn.functional.pad(tmp_image, (0, 0, 0, 0, 0, 0, 0, padding_size), "constant", 0)
                            image_data.append(tmp_image)
                    else:
                        image_data.append(default_collate(single_data['image']))
                elif 'aspect_ratio_ids' in data[0].keys():
                    aspect_ratio_ids.append(torch.LongTensor(default_collate(single_data['aspect_ratio_ids'])))
                    aspect_ratio_mask.append(torch.LongTensor(default_collate(single_data['aspect_ratio_mask'])))

                    image_data.append(default_collate(single_data['image']))
                else:
                    image_data.append(default_collate(single_data['image']))
                
                image_num.append(single_data['image_num'])
            try:
                query_ids.append(single_data['query_id'])
            except:
                pass

        if 'image_sizes' in data[0].keys():
            image_sizes = torch.concat(image_sizes, dim=0)
            batch['image_sizes'] = image_sizes

        if 'aspect_ratio_ids' in data[0].keys():
            torch.concat(aspect_ratio_ids, dim=-1).size()
            aspect_ratio_ids = torch.concat(aspect_ratio_ids, dim=-2).squeeze(0)
            aspect_ratio_mask = torch.concat(aspect_ratio_mask, dim=-3).squeeze(0)
            
            batch['aspect_ratio_ids'] = aspect_ratio_ids
            batch['aspect_ratio_mask'] = aspect_ratio_mask

            image = torch.concat(image_data, dim=0)
        else:
            image = torch.concat(image_data, dim=0)
    
        batch['input_ids'] = input_ids
        batch['labels'] = labels
        batch['attention_mask'] = attention_mask
        batch['image'] = image
        batch['image_num'] = image_num
        batch['query_id'] = query_ids
        return batch

class DataCollatorPadToMaxLenForPPOTraining:

    def __init__(self, max_token_len, pad_token_id, image_size):
        self.max_token_len = max_token_len
        self.pad_token_id = pad_token_id
        self.image_size = image_size

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

        image_num = []
        image_data = []
        image_sizes = []
        aspect_ratio_ids = []
        aspect_ratio_mask = []
        for single_data in data:
            if single_data['image'][0] is None:
                if 'image_sizes' in single_data.keys():
                    image_data.append(torch.zeros(1, 5, 3, self.image_size['height'],self.image_size['width']))
                    image_sizes.append(torch.LongTensor([123, 123]))
                elif 'aspect_ratio_ids' in data[0].keys():
                    image_data.append(torch.zeros(1, 4, 3, self.image_size['height'],self.image_size['width']))
                    aspect_ratio_ids.append(torch.LongTensor([[4]]))
                    aspect_ratio_mask.append(torch.LongTensor([[[0,0,0,0]]]))
                else:
                    image_data.append(torch.zeros(1, 3, self.image_size['height'],self.image_size['width']))
                
                image_num.append(0)
            else:
                if 'image_sizes' in single_data.keys():
                    if len(single_data['image_sizes']) != 0:
                        image_sizes.append(default_collate(single_data['image_sizes']))
                        if default_collate(single_data['image']).size(1) == 5:
                            image_data.append(default_collate(single_data['image']))
                        else:
                            tmp_image = default_collate(single_data['image'])
                            current_dim_size = tmp_image.size(1)
                            padding_size = 5 - current_dim_size
                            tmp_image = torch.nn.functional.pad(tmp_image, (0, 0, 0, 0, 0, 0, 0, padding_size), "constant", 0)
                            image_data.append(tmp_image)
                    else:
                        image_data.append(default_collate(single_data['image']))

                elif 'aspect_ratio_ids' in data[0].keys():
                    aspect_ratio_ids.append(torch.LongTensor(default_collate(single_data['aspect_ratio_ids'])))
                    aspect_ratio_mask.append(torch.LongTensor(default_collate(single_data['aspect_ratio_mask'])))
                    image_data.append(default_collate(single_data['image']))
                else:
                    image_data.append(default_collate(single_data['image']))
                
                image_num.append(single_data['image_num'])

        image = torch.concat(image_data, dim=0)

        if 'image_sizes' in data[0].keys():
            image_sizes = torch.concat(image_sizes, dim=0)
            batch['image_sizes'] = image_sizes

        if 'aspect_ratio_ids' in data[0].keys():
            aspect_ratio_ids = torch.concat(aspect_ratio_ids, dim=0)
            aspect_ratio_mask = torch.concat(aspect_ratio_mask, dim=0)

            batch['aspect_ratio_ids'] = aspect_ratio_ids
            batch['aspect_ratio_mask'] = aspect_ratio_mask

        image_num = [f['image_num'] for f in data]
        batch['input_ids'] = input_ids
        batch['labels'] = labels
        batch['attention_mask'] = attention_mask
        batch['image'] = image
        batch['image_num'] = image_num
        return batch

class DataCollatorPadToMaxLenForPrediction:

    def __init__(self, max_token_len, pad_token_id, image_size):
        self.max_token_len = max_token_len
        self.pad_token_id = pad_token_id
        self.image_size = image_size

    def __call__(self, data):
        batch = {}
 
        input_ids = data[0]['input_ids']
        labels = data[0]['labels']
        attention_mask = data[0]['attention_mask']
        sample_id = data[0]['id']

        if len(data[0]['image']) == 0:
            if 'image_sizes' in data[0].keys():
                image_data.append(torch.zeros(1, 5, 3, self.image_size['height'],self.image_size['width']))
                image_sizes = []
                image_sizes.append(torch.LongTensor([123, 123]))
            elif 'aspect_ratio_ids' in data[0].keys():
                image_data.append(torch.zeros(1, 4, 3, self.image_size['height'],self.image_size['width']))
                aspect_ratio_ids = []
                aspect_ratio_mask = []
                aspect_ratio_ids.append(torch.LongTensor([[4]]))
                aspect_ratio_mask.append(torch.LongTensor([[[0,0,0,0]]]))
            else:
                image_data.append(torch.zeros(1, 3, self.image_size['height'],self.image_size['width']))
            image_num = [0]
        else:
            image_num = []
            image_data = []
            image_sizes = []
            aspect_ratio_ids = []
            aspect_ratio_mask = []
            
            for single_data in data:
                if 'image_sizes' in single_data.keys():
                    if len(single_data['image_sizes']) != 0:
                        image_sizes.append(torch.LongTensor(default_collate(single_data['image_sizes'])))
                
                if 'aspect_ratio_ids' in single_data.keys():
                    aspect_ratio_ids.append(torch.LongTensor(default_collate(single_data['aspect_ratio_ids'])))
                    aspect_ratio_mask.append(torch.LongTensor(default_collate(single_data['aspect_ratio_mask'])))

                image_data.append(default_collate(single_data['image'][0]))
                    
                image_num.append(single_data['image_num'])

        image = torch.concat(image_data, dim=0)
        
        if 'image_sizes' in data[0].keys():
            image_sizes = torch.concat(image_sizes, dim=0)
            batch['image_sizes'] = image_sizes

        if 'aspect_ratio_ids' in data[0].keys():
            aspect_ratio_ids = torch.concat(aspect_ratio_ids, dim=0)
            aspect_ratio_mask = torch.concat(aspect_ratio_mask, dim=0)

            batch['aspect_ratio_ids'] = aspect_ratio_ids
            batch['aspect_ratio_mask'] = aspect_ratio_mask

        batch['input_ids'] = torch.LongTensor(input_ids).unsqueeze(0)
        batch['labels'] = torch.LongTensor(labels).unsqueeze(0)
        batch['attention_mask'] = torch.LongTensor(attention_mask).unsqueeze(0)
        batch['image'] = image
        batch['image_num'] = image_num
        batch['id'] = sample_id
        return batch