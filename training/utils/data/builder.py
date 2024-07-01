import numpy as np
import torch

from .llava_dataset import LlavaDataset, LlavaComparsionDataset, LlavaPPODataset, LlavaPredictDataset  
from .vqa_dataset import ConcatDataset
from utils.utils import print_rank_0


def build_dataset(data_path, data_debug_path, dataset_name, dataset_sample,
                  dataset_concatenate_samples, max_num_image_per_sample, 
                  vis_root=None, **kwargs):
    if isinstance(dataset_name, list):
        datasets = [build_dataset(data_path, data_debug_path,
                                  dataset_name[i], dataset_sample[i],
                                  dataset_concatenate_samples[i],
                                  max_num_image_per_sample,
                                  vis_root=vis_root,
                                  **kwargs) for i in range(len(dataset_name))]
        return ConcatDataset(datasets)
    
    if dataset_name == "llava_reward":
        dataset = LlavaComparsionDataset(
            data_path,
            data_debug_path,
            dataset_concatenate_samples,
            vis_root=vis_root,
            **kwargs,
        )
    elif dataset_name == "llava_ppo":
        dataset = LlavaPPODataset(
            data_path,
            data_debug_path,
            dataset_concatenate_samples,
            vis_root=vis_root,
            **kwargs,
        )
    elif dataset_name == "llava_predict":
        dataset = LlavaPredictDataset(
            data_path,
            data_debug_path,
            dataset_concatenate_samples,
            vis_root=vis_root,
            **kwargs,
        )
    elif dataset_name == "llava_sft":
        dataset = LlavaDataset(
            data_path,
            data_debug_path,
            dataset_concatenate_samples,
            vis_root=vis_root,
            **kwargs,
        )
    else:
        raise NotImplementedError

    if dataset_sample != 'all':
        dataset_sample = int(dataset_sample)
        random_indices = np.random.choice(len(dataset), min(dataset_sample, len(dataset)), replace=False)
        subsample_dataset = torch.utils.data.Subset(dataset, random_indices)
        subsample_dataset.collater = dataset.collater
        print_rank_0(f"[DATA] Built dataset {dataset_name} with {len(subsample_dataset)} samples.")
        return subsample_dataset
    else:
        print_rank_0(f"[DATA] Built dataset {dataset_name} with all {len(dataset)} samples.")
        return dataset
