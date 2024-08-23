# This file is adapted from https://github.com/open-mmlab/Multimodal-GPT
# This dataset is from https://llava-vl.github.io/
import os
import copy
import training.utils.data.DST as DST
from .vqa_dataset import VQADataset
from training.utils.utils import get_rank
from .utils import save_debug_text


import torch
import numpy as np

class LlavaDataset(VQADataset):
    def __init__(self, data_path, data_debug_path, per_sample_image, tokenizer, vis_processor, vis_root, **kwargs):
        assert os.path.isdir(vis_root), f"LlavaDataset image directory {vis_root} not found, you need to download 2017 Train images from https://cocodataset.org/#download"
        ann_paths = [data_path]
        for idx in range(len(ann_paths)):
            assert os.path.isfile(ann_paths[idx]), f"LlavaDataset annotation file {ann_paths[idx]} not found, you need to download it from https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K"
        super().__init__(data_path, data_debug_path, per_sample_image, tokenizer, vis_processor,
                         vis_root, ann_paths, **kwargs)

    def _add_instance_ids(self, key="id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)

    def process_text(self, ann, data_debug_path=None, data_debug_counter=0, first_message=False):
        num_convs = len(ann["conversations"]) // 2
        conv_list = []
        
        for conv_idx in range(num_convs):

            with_image_flag = False
            question = ann["conversations"][conv_idx*2]["value"]
            if conv_idx == 0: 
                # the first turn, image = True
                with_image_flag = True
                # question = question.replace("<image>","").replace("\n","")
                question = question.replace("<image>", "").strip("\n")

            end_of_token = ""
            if self.template == "llama_3":
                end_of_token = DST.LLAMA3_HUMAN_QUESTION_PRETOKEN_END
            elif self.template == "llama_2":
                end_of_token = DST.LLAMA2_HUMAN_QUESTION_PRETOKEN_END
            elif self.template == "vicuna":
                end_of_token = DST.VICUNA_HUMAN_QUESTION_PRETOKEN_END

            answer = ann["conversations"][conv_idx*2+1]["value"] + end_of_token
            instruction = self.prompter(question, with_image=with_image_flag, first_message=(len(conv_list) == 0 and first_message), template=self.template)
            single_conv = dict(instruction=instruction, answer=answer)
            conv_list.append(single_conv)
        
        save_debug_text(conv_list, 
                        data_debug_path, 
                        data_debug_counter,
                        get_rank()
                        )
        return conv_list

    def __getitem__(self, index):
        
        res_list_all = []
        for ann in self.annotation[index]:
            if 'image' in ann.keys():
                if ann['image'] is not None:
                    image = self.process_image(
                        ann,
                        data_debug_path=self.data_debug_path,
                        data_debug_counter=self.data_debug_counter
                    )
                else:
                    image = None
            else:
                image = None
            text_list = self.process_text(
                ann,
                data_debug_path=self.data_debug_path,
                data_debug_counter=self.data_debug_counter,
                first_message=True
            )
            self.data_debug_counter += 1
            res_list = []
            for text in text_list:
                single_res = self.tokenize(text)
                res_list.append(single_res)
            input_ids = []
            attention_mask = []
            labels = []
            length_of_res = len(res_list)
            if length_of_res == 1:
                # single turn
                res = res_list[0]
                input_ids.extend(res["input_ids"])
                attention_mask.extend(res["attention_mask"])
                labels.extend(res["labels"])
            else:
                # multi-turn
                for idx in range(length_of_res):
                    res = res_list[idx]
                    if idx == 0:
                        # delete the eos
                        if (res["input_ids"][-1]==self.tokenizer.eos_token_id) :
                            res["input_ids"] = res["input_ids"][0:-1]
                            res["attention_mask"] = res["attention_mask"][0:-1]
                            res["labels"] = res["labels"][0:-1]

                    elif idx == (length_of_res-1):
                        # delete the bos
                        if (res["input_ids"][0]==self.tokenizer.bos_token_id):
                            res["input_ids"] = res["input_ids"][1:]
                            res["attention_mask"] = res["attention_mask"][1:]
                            res["labels"] = res["labels"][1:]

                    else:
                        # delete eos&bos
                        if (res["input_ids"][-1]==self.tokenizer.eos_token_id) :
                            res["input_ids"] = res["input_ids"][0:-1]
                            res["attention_mask"] = res["attention_mask"][0:-1]
                            res["labels"] = res["labels"][0:-1]

                        if (res["input_ids"][0]==self.tokenizer.bos_token_id):
                            res["input_ids"] = res["input_ids"][1:]
                            res["attention_mask"] = res["attention_mask"][1:]
                            res["labels"] = res["labels"][1:]
                        
                    input_ids.extend(res["input_ids"])
                    attention_mask.extend(res["attention_mask"])
                    labels.extend(res["labels"])
            res = dict(
                input_ids = input_ids,
                attention_mask = attention_mask,
                labels = labels,
                image=image
            )
            res_list_all.append(res)
        output = self.merge_all_images(res_list_all, text)
        return output


class LlavaComparsionDataset(VQADataset):
    def __init__(self, data_path, data_debug_path, per_sample_image, tokenizer, vis_processor, vis_root=None, max_ranked_candidate_num=4, **kwargs):
        self.max_cand_num = max_ranked_candidate_num
        assert os.path.isdir(vis_root), f"LlavaDataset image directory {vis_root} not found, you need to download 2017 Train images from https://cocodataset.org/#download"
        ann_paths = [data_path]
        for idx in range(len(ann_paths)):
            assert os.path.isfile(ann_paths[idx]), f"LlavaDataset annotation file {ann_paths[idx]} not found, you need to download it from https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K"
        per_sample_image = 1
        super().__init__(data_path, data_debug_path, per_sample_image, tokenizer, vis_processor,
                         vis_root, ann_paths, **kwargs)

    def _add_instance_ids(self, key="id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)

    def __getitem__(self, index):
        outputs = []
        res_list = []
        for ann in self.annotation[index]:
            if ann['image'] != None:
                image = self.process_image(ann,
                                        data_debug_path=self.data_debug_path,
                                        data_debug_counter=self.data_debug_counter)
            else:
                image = None
            
            text = self.process_text(ann,
                                    data_debug_path=self.data_debug_path,
                                    data_debug_counter=self.data_debug_counter,
                                    first_message=(not res_list))
            self.data_debug_counter += 1

            ranked_candidates = text['answer']
            query_id = [ann["id"]]

            
            if len(ranked_candidates) > len(ranked_candidates):
                pass
            else:
                candi_num = len(ranked_candidates)
                for rc in ranked_candidates:
                    res_list = []
                    text_tmp = {'instruction': text['instruction'], 'answer': rc}
                    res = self.tokenize(text_tmp)
                    res.update(image=image)
                    res.update(text_tmp)
                    res.update(query_id=query_id)
                    res_list.append(res)
                    output = self.merge_all_images(res_list, text)
                    outputs.append(output)
                for i in range(self.max_cand_num - candi_num):
                    tmp = {
                        "input_ids": [0 for item in outputs[0]['input_ids']],
                        "attention_mask": [0 for item in outputs[0]['attention_mask']],
                        "labels": [0 for item in outputs[0]['labels']],
                        "score": [],
                        "image": [None for item in outputs[0]['image']],
                        "image_num": 0
                    }
                    tmp["input_ids"][0] = -1
                    outputs.append(tmp)
        return outputs

    def process_text(self, ann, data_debug_path=None, data_debug_counter=0, first_message=False):
        question = ann["conversations"][0]["value"]
        # remove '<image>' tag and '\n'
        # question = question.replace("<image>", "").replace("\n", "")
        question = question.replace("<image>", "")
        question = question.strip("USER:").strip("ASSISTANT:")
        question = question.strip("\n")
        question = question.strip(" ")

        end_of_token = ""
        if self.template == "llama_3":
            end_of_token = DST.LLAMA3_HUMAN_QUESTION_PRETOKEN_END
        elif self.template == "llama_2":
            end_of_token = DST.LLAMA2_HUMAN_QUESTION_PRETOKEN_END
        elif self.template == "vicuna":
            end_of_token = DST.VICUNA_HUMAN_QUESTION_PRETOKEN_END
        
        for idx in range(len(ann["conversations"][1]["value"])):
            ann["conversations"][1]["value"][idx] += end_of_token

        answer = ann["conversations"][1]["value"]
        instruction = self.prompter(question, with_image=True, first_message=first_message, template=self.template)
        
        save_debug_text([instruction, answer], data_debug_path, data_debug_counter, get_rank())
        
        return dict(instruction=instruction, answer=answer)

class LlavaRMPaddingDataset(VQADataset):
    def __init__(self, data_path, data_debug_path, per_sample_image, tokenizer, vis_processor, vis_root=None, **kwargs):
        assert os.path.isdir(vis_root), f"LlavaDataset image directory {vis_root} not found, you need to download 2017 Train images from https://cocodataset.org/#download"
        ann_paths = [data_path]
        for idx in range(len(ann_paths)):
            assert os.path.isfile(ann_paths[idx]), f"LlavaDataset annotation file {ann_paths[idx]} not found, you need to download it from https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K"
        per_sample_image = 1
        super().__init__(data_path, data_debug_path, per_sample_image, tokenizer, vis_processor,
                         vis_root, ann_paths, **kwargs)

    def _add_instance_ids(self, key="id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)

    def __getitem__(self, index):
        outputs = []
        res_list = []
        for ann in self.annotation[index]:
            if ann['image'] != None:
                image = self.process_image(ann,
                                        data_debug_path=self.data_debug_path,
                                        data_debug_counter=self.data_debug_counter)
            else:
                image = None
            
            text = self.process_text(ann,
                                    data_debug_path=self.data_debug_path,
                                    data_debug_counter=self.data_debug_counter,
                                    first_message=(not res_list))
            self.data_debug_counter += 1

            ranked_candidates = text['answer']

            self.max_cand_num = 4
            if len(ranked_candidates) > self.max_cand_num:
                pass
            else:
                candi_num = len(ranked_candidates)
                for rc in ranked_candidates:
                    res_list = []
                    text_tmp = {'instruction': text['instruction'], 'answer': rc}
                    res = self.tokenize(text_tmp)
                    res.update(image=image)
                    res.update(text_tmp)
                    res_list.append(res)
                    output = self.merge_all_images(res_list, text)
                    outputs.append(output)
                for i in range(self.max_cand_num - candi_num):
                    # import pdb; pdb.set_trace()

                    tmp = {
                        "input_ids": [0 for item in outputs[0]['input_ids']],
                        "attention_mask": [0 for item in outputs[0]['attention_mask']],
                        "labels": [0 for item in outputs[0]['labels']],
                        "score": [],
                        "image": [None for item in outputs[0]['image']],
                        "image_num": 0
                    }
                    tmp["input_ids"][0] = -1
                    outputs.append(tmp)
        return outputs

    def process_text(self, ann, data_debug_path=None, data_debug_counter=0, first_message=False):
        question = ann["conversations"][0]["value"]
        # remove '<image>' tag and '\n'
        # question = question.replace("<image>", "").replace("\n", "")
        question = question.replace("<image>", "")
        question = question.strip("USER:").strip("ASSISTANT:")
        question = question.strip("\n")
        question = question.strip(" ")

        end_of_token = ""
        if self.template == "llama_3":
            end_of_token = DST.LLAMA3_HUMAN_QUESTION_PRETOKEN_END
        elif self.template == "llama_2":
            end_of_token = DST.LLAMA2_HUMAN_QUESTION_PRETOKEN_END
        elif self.template == "vicuna":
            end_of_token = DST.VICUNA_HUMAN_QUESTION_PRETOKEN_END
        
        for idx in range(len(ann["conversations"][1]["value"])):
            ann["conversations"][1]["value"][idx] += end_of_token

        answer = ann["conversations"][1]["value"]
        instruction = self.prompter(question, with_image=True, first_message=first_message, template=self.template)
        
        save_debug_text([instruction, answer], data_debug_path, data_debug_counter, get_rank())
        
        return dict(instruction=instruction, answer=answer)



class LlavaRewardMseDataset(VQADataset):
    def __init__(self, data_path, data_debug_path, per_sample_image, tokenizer, vis_processor, vis_root=None, **kwargs):
        assert os.path.isdir(vis_root), f"LlavaDataset image directory {vis_root} not found, you need to download 2017 Train images from https://cocodataset.org/#download"
        ann_paths = [data_path]
        for idx in range(len(ann_paths)):
            assert os.path.isfile(ann_paths[idx]), f"LlavaDataset annotation file {ann_paths[idx]} not found, you need to download it from https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K"
        per_sample_image = 1
        super().__init__(data_path, data_debug_path, per_sample_image, tokenizer, vis_processor,
                         vis_root, ann_paths, **kwargs)

    def _add_instance_ids(self, key="id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)

    def __getitem__(self, index):
        outputs = []
        res_list = []
        for ann in self.annotation[index]:
            if ann['image'] != None:
                image = self.process_image(ann,
                                        data_debug_path=self.data_debug_path,
                                        data_debug_counter=self.data_debug_counter)
            else:
                image = None
            
            text = self.process_text(ann,
                                    data_debug_path=self.data_debug_path,
                                    data_debug_counter=self.data_debug_counter,
                                    first_message=(not res_list))
            self.data_debug_counter += 1

            ranked_candidates = text['answer']
            score = [ann['score']]
            rc = ranked_candidates[0]
            res_list = []
            text_tmp = {'instruction': text['instruction'], 'answer': rc}
            res = self.tokenize(text_tmp)
            res.update(image=image)
            res.update(text_tmp)
            res.update(score=score)
            res_list.append(res)
            output = self.merge_all_images(res_list, text)
            outputs.append(output)
    
        return outputs

    def process_text(self, ann, data_debug_path=None, data_debug_counter=0, first_message=False):
        question = ann["conversations"][0]["value"]
        # remove '<image>' tag and '\n'
        # question = question.replace("<image>", "").replace("\n", "")
        question = question.replace("<image>", "")
        question = question.strip("USER:").strip("ASSISTANT:")
        question = question.strip("\n")
        question = question.strip(" ")

        end_of_token = ""
        if self.template == "llama_3":
            end_of_token = DST.LLAMA3_HUMAN_QUESTION_PRETOKEN_END
        elif self.template == "llama_2":
            end_of_token = DST.LLAMA2_HUMAN_QUESTION_PRETOKEN_END
        elif self.template == "vicuna":
            end_of_token = DST.VICUNA_HUMAN_QUESTION_PRETOKEN_END
        
        for idx in range(len(ann["conversations"][1]["value"])):
            ann["conversations"][1]["value"][idx] += end_of_token

        answer = ann["conversations"][1]["value"]
        instruction = self.prompter(question, with_image=True, first_message=first_message, template=self.template)
        
        save_debug_text([instruction, answer], data_debug_path, data_debug_counter, get_rank())
        
        return dict(instruction=instruction, answer=answer)

class LlavaPPODataset(VQADataset):
    def __init__(self, data_path, data_debug_path, per_sample_image, tokenizer, vis_processor, vis_root=None, **kwargs):
        assert os.path.isdir(vis_root), f"LlavaDataset image directory {vis_root} not found, you need to download 2017 Train images from https://cocodataset.org/#download"
        ann_paths = [data_path]
        for idx in range(len(ann_paths)):
            assert os.path.isfile(ann_paths[idx]), f"LlavaDataset annotation file {ann_paths[idx]} not found, you need to download it from https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K"
        per_sample_image = 1
        super().__init__(data_path, data_debug_path, per_sample_image, tokenizer, vis_processor,
                         vis_root, ann_paths, add_eos=False, **kwargs)

    def _add_instance_ids(self, key="id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)

    def tokenize(self, text):
        res = self.tokenizer(
            text["instruction"],
            return_tensors=None,
            padding="do_not_pad",
            truncation=True,
            max_length=512,
        )
        if (res["input_ids"][-1] == self.tokenizer.eos_token_id) and (not self.add_eos):
            res["input_ids"] = res["input_ids"][0:-1]
            res["attention_mask"] = res["attention_mask"][0:-1]

        labels = copy.deepcopy(res["input_ids"])
        # ignore instruction_token
        if self.ignore_instruction:
            instruction_token = self.tokenizer(
                text["instruction"], return_tensors=None, padding="do_not_pad", truncation=True, max_length=512
            )
            labels = [DST.DEFAULT_LABEL_PADDING_NUM] * len(instruction_token["input_ids"]) + labels[len(instruction_token["input_ids"]) :]

        res.update(labels=labels)
        return res

    def __getitem__(self, index):
        res_list = []
        for ann in self.annotation[index]:
            if ann['image'] != None:
                if self.template == 'llava_next':
                    image, image_sizes = self.process_image(ann,
                                        data_debug_path=self.data_debug_path,
                                        data_debug_counter=self.data_debug_counter)
                else:
                    image = self.process_image(ann,
                                            data_debug_path=self.data_debug_path,
                                            data_debug_counter=self.data_debug_counter)
                with_image = True
            else:
                image = None
                with_image = False
            
            text = self.process_text(ann,
                                    data_debug_path=self.data_debug_path,
                                    data_debug_counter=self.data_debug_counter,
                                    first_message=True,
                                    with_image=with_image)

            self.data_debug_counter += 1
            res = self.tokenize(text)
            if self.template == 'llava_next':
                res.update(image_sizes=image_sizes)
            res.update(image=image)
            res.update(text)
            res_list.append(res)
        
        output = self.merge_all_images(res_list, text)
        return output

    def process_text(self, ann, data_debug_path=None, data_debug_counter=0, first_message=False, with_image=False):
        question = ann["conversations"][0]["value"]
        # remove '<image>' tag and '\n'
        # question = question.replace("<image>", "").replace("\n", "")
        question = question.replace("<image>", "").strip("\n")
        end_of_token = ""
        if self.template == "llama_3":
            end_of_token = DST.LLAMA3_HUMAN_QUESTION_PRETOKEN_END
        elif self.template == "llama_2":
            end_of_token = DST.LLAMA2_HUMAN_QUESTION_PRETOKEN_END
        elif self.template == "vicuna":
            end_of_token = DST.VICUNA_HUMAN_QUESTION_PRETOKEN_END

        answer = ann["conversations"][1]["value"] + end_of_token
        instruction = self.prompter(question, with_image=with_image, first_message=first_message, template=self.template)
        
        save_debug_text([instruction, answer], data_debug_path, data_debug_counter, get_rank())
        
        return dict(instruction=instruction, answer=answer)
    
class LlavaPredictDataset(VQADataset):
    def __init__(self, data_path, data_debug_path, per_sample_image, tokenizer, vis_processor, vis_root=None, **kwargs):
        assert os.path.isdir(vis_root), f"LlavaDataset image directory {vis_root} not found, you need to download 2017 Train images from https://cocodataset.org/#download"
        ann_paths = [data_path]
        per_sample_image = 1
        super().__init__(data_path, data_debug_path, per_sample_image, tokenizer, vis_processor,
                         vis_root, ann_paths, add_eos=False, **kwargs)

    def _add_instance_ids(self, key="id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)

    def tokenize(self, text):
        res = self.tokenizer(
            text["instruction"],
            return_tensors=None,
            padding="do_not_pad",
            truncation=True,
            max_length=512,
        )
        if (res["input_ids"][-1] == self.tokenizer.eos_token_id) and (not self.add_eos):
            res["input_ids"] = res["input_ids"][0:-1]
            res["attention_mask"] = res["attention_mask"][0:-1]

        labels = copy.deepcopy(res["input_ids"])
        # ignore instruction_token
        if self.ignore_instruction:
            instruction_token = self.tokenizer(
                text["instruction"], return_tensors=None, padding="do_not_pad", truncation=True, max_length=512
            )
            labels = [DST.DEFAULT_LABEL_PADDING_NUM] * len(instruction_token["input_ids"]) + labels[len(instruction_token["input_ids"]) :]

        res.update(labels=labels)
        return res
    
    def __getitem__(self, index):
        res_list = []
        for ann in self.annotation[index]:
            if self.template == 'llava_next':
                image, image_sizes = self.process_image(ann,
                                    data_debug_path=self.data_debug_path,
                                    data_debug_counter=self.data_debug_counter)
            else:
                image = self.process_image(ann,
                                        data_debug_path=self.data_debug_path,
                                        data_debug_counter=self.data_debug_counter)
            with_image = True

            text = self.process_text(ann,
                                    data_debug_path=self.data_debug_path,
                                    data_debug_counter=self.data_debug_counter,
                                    first_message=True,
                                    with_image=with_image)
     
            self.data_debug_counter += 1
            res = self.tokenize(text)
            if self.template == 'llava_next':
                res.update(image_sizes=image_sizes)
            res.update(image=image)
            res.update(text)
            id_dict = {
                "id": ann['id']
            }
            res.update(id_dict)

            res_list.append(res)
        
        output = self.merge_all_images(res_list, text)
        return output
    
    def merge_all_images(self, res_list, text):
        image_number = 0 
        original_output = {"input_ids": [], "attention_mask": [], "labels": [], "image": [], "id": []} #copy.deepcopy(self.system_instruct)
        for res in res_list:
            original_output["input_ids"] = original_output["input_ids"] + res["input_ids"]
            original_output["attention_mask"] = original_output["attention_mask"] + res["attention_mask"]
            original_output["labels"] = original_output["labels"] + res["labels"]
            original_output['id'] = original_output['id'] + [res['id']]
            if DST.DEFAULT_IMAGE_TOKEN in text["instruction"]:
                image_number += 1
                original_output["image"] = original_output["image"] + [res["image"]]

                if "image_sizes" in res.keys():
                    original_output.update(image_sizes=[res["image_sizes"]])

        if image_number == 0:
            print("Warning: Here is input without image")

        original_output["image_num"] = image_number
        return original_output

    def process_text(self, ann, data_debug_path=None, data_debug_counter=0, first_message=False, with_image=False):
        question = ann["conversations"][0]["value"]
        question = question.replace("<image>", "").strip("\n")
        end_of_token = ""

        if self.template == "llama_3":
            end_of_token = DST.LLAMA3_HUMAN_QUESTION_PRETOKEN_END
        elif self.template == "llama_2":
            end_of_token = DST.LLAMA2_HUMAN_QUESTION_PRETOKEN_END
        elif self.template == "vicuna":
            end_of_token = DST.VICUNA_HUMAN_QUESTION_PRETOKEN_END

        answer = ann["conversations"][1]["value"] + end_of_token

        instruction = self.prompter(question, with_image=with_image, first_message=first_message, template=self.template)
        
        save_debug_text([instruction, answer], data_debug_path, data_debug_counter, get_rank())
        
        return dict(instruction=instruction, answer=answer)

    def collater(self, samples):
        image_list, question_list, answer_list, input_id_list, attention_mask_list, labels_list, id_list = [], [], [], [], [], [],[]

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["instruction"])
            answer_list.append(sample["answer"])
            input_id_list.append(sample["input_ids"])
            attention_mask_list.append(sample["attention_mask"])
            labels_list.append(sample["labels"])
            id_list.append(sample['id'])

        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        max_label_length = max(len(l) for l in labels_list)
        padding_side = self.tokenizer.padding_side
        padded_labels = []
        for l in labels_list:
            remainder = [DST.DEFAULT_LABEL_PADDING_NUM] * (max_label_length - len(l))
            if isinstance(l, list):
                l = l + remainder if padding_side == "right" else remainder + l
            elif padding_side == "right":
                l = np.concatenate([l, remainder]).astype(np.int64)
            else:
                l = np.concatenate([remainder, l]).astype(np.int64)
            padded_labels.append(l)

        padded_samples = self.tokenizer.pad(
            {"input_ids": input_id_list, "attention_mask": attention_mask_list, "labels": padded_labels},
            return_tensors="pt",
            padding="longest",
        )

        # remove all image related tokens
        labels = padded_samples["labels"]
        labels[labels == self.tokenizer.pad_token_id] = DST.DEFAULT_LABEL_PADDING_NUM
        labels[:, 0] = DST.DEFAULT_LABEL_PADDING_NUM
        for k, v in self.image_token_dict.items():
            labels[labels == v] = DST.DEFAULT_LABEL_PADDING_NUM
        return {
            "image": torch.stack(image_list, dim=0),
            "input_ids": padded_samples["input_ids"],
            "attention_mask": padded_samples["attention_mask"],
            "labels": labels,
            "instruction": question_list,
            "answer": answer_list,
            'id': id_list
        }

class LlavaRewardEvalDataset(LlavaPredictDataset):

    def __init__(self, data_path, data_debug_path, per_sample_image, tokenizer, vis_processor, vis_root=None, **kwargs):
        super().__init__(data_path, data_debug_path, per_sample_image, tokenizer, vis_processor, vis_root, **kwargs)

    def tokenize(self, text):
        res = self.tokenizer(
            text["instruction"] + text["answer"],
            return_tensors=None,
            padding="do_not_pad",
            truncation=True,
            max_length=512,
        )

        labels = copy.deepcopy(res["input_ids"])
        # ignore instruction_token
        if self.ignore_instruction:
            instruction_token = self.tokenizer(
                text["instruction"], return_tensors=None, padding="do_not_pad", truncation=True, max_length=512
            )
            labels = [DST.DEFAULT_LABEL_PADDING_NUM] * len(instruction_token["input_ids"]) + labels[len(instruction_token["input_ids"]) :]

        res.update(labels=labels)
        return res
    
    