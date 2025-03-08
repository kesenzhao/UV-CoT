import io
import os
import json
import tqdm
import copy
import torch
import itertools
import pandas as pd
import torch.utils.data as torch_data
import PIL.Image as PIL_image
from functools import partial
from muffin.train.train_utils import encode_multimodal_preference_sample, SFT_collator_fn, preprocess_v1


def bytes_to_PIL_image(img_buffer):
    img_io = io.BytesIO(img_buffer)
    img_io.seek(0)
    # print('img_io:', img_io)
    # print('img_buffer:', img_buffer)
    image = PIL_image.open(img_io).convert('RGB')

    return image

def get_batch_logps_minicpm(logits: torch.FloatTensor, labels: torch.LongTensor, return_per_token_logp=False, return_all=False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, :-1].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2,
                                   index=labels.unsqueeze(2)).squeeze(2)

    log_prob = (per_token_logps * loss_mask).sum(-1)
    average_log_prob = log_prob / loss_mask.sum(-1)

    assert per_token_logps.shape == labels.shape, f"per_token_logps.shape={per_token_logps.shape}, labels.shape={labels.shape}"
    if return_per_token_logp:
        return per_token_logps

    if return_all:
        return per_token_logps, log_prob, average_log_prob

    return log_prob, average_log_prob


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, return_per_token_logp=False, return_all=False, tokenizer=None) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape, f'logits.shape[:-1]={logits.shape[:-1]}, labels.shape={labels.shape}'

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2,
                                   index=labels.unsqueeze(2)).squeeze(2)

    log_prob = (per_token_logps * loss_mask).sum(-1)
    average_log_prob = log_prob / loss_mask.sum(-1)

    # print("==>", labels)

    # print(per_token_logps.shape, labels.shape)
    if return_per_token_logp:
        return per_token_logps

    if return_all:
        return per_token_logps, log_prob, average_log_prob

    return log_prob, average_log_prob



def compute_cross_similarity_matrix(scaled_similarity, token_types):
    """
    计算图像和文本 token 之间的交叉相似性，通过矩阵化操作实现。

    参数：
    - scaled_similarity (Tensor): 形状为 [batch_size, seq_len, seq_len] 的相似性矩阵。
    - token_types (Tensor): 形状为 [batch_size, seq_len] 的 token 类型，其中 0 表示文本 token，1 表示图像 token。

    返回：
    - cross_similarity_scores (Tensor): 形状为 [batch_size, num_image_tokens]，每个图像 token 与所有文本 token 之间的交叉相似性之和。
    """
    batch_size, seq_len, _ = scaled_similarity.shape

    # 创建文本和图像 token 的 mask
    text_mask = token_types == 0  # [batch_size, seq_len]
    image_mask = token_types == 1  # [batch_size, seq_len]

    # 通过广播和矩阵运算，获取图像 token 和文本 token 之间的相似性
    text_indices = text_mask.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, seq_len]
    image_indices = image_mask.unsqueeze(2).expand(-1, seq_len, -1)  # [batch_size, seq_len, seq_len]

    # 计算每个图像 token 与所有文本 token 之间的相似性
    cross_similarity_scores = (scaled_similarity * image_indices * text_indices).sum(dim=-1)  # [batch_size, seq_len]

    # 选择每个 batch 中的图像 token 对应的交叉相似性
    # image_cross_similarity_scores = cross_similarity_scores.gather(1, image_mask.sum(dim=-1, keepdim=True) - 1)
    image_token_indices = image_mask.nonzero(as_tuple=True)  # 返回的是(batch_size, seq_len)维度上的图像token位置索引

    # 获取每个图像token的相似性分数
    # 这里我们使用gather来根据图像token的索引从cross_similarity_scores中选取对应的相似性
    # print(image_token_indices[1].shape)
    # image_cross_similarity_scores = torch.gather(cross_similarity_scores, dim=1, index=image_token_indices[1].unsqueeze(1))
    # print('token_indices:', image_token_indices.shape)
    # print('similarity_scores:', cross_similarity_scores.shape)
    image_cross_similarity_scores = cross_similarity_scores[image_token_indices].view(batch_size, -1)   # [batch_size, image_token_num]

    # # 调整维度，确保输出为 [batch_size, image_token_num]
    # print('image_cross_similarity_scores', image_cross_similarity_scores.shape)
    # image_cross_similarity_scores = image_cross_similarity_scores.squeeze(1) # [batch_size, image_token_num]    
    return image_cross_similarity_scores



def get_batch_logps_cot(logits: torch.FloatTensor, labels: torch.LongTensor, CoT_labels, token_types, return_per_token_logp=False, return_all=False, tokenizer=None) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape, f'logits.shape[:-1]={logits.shape[:-1]}, labels.shape={labels.shape}'
    # print('shape:', logits.shape, labels.shape)
    labels = labels[:, 1:].clone()
    token_types = token_types[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2,
                                   index=labels.unsqueeze(2)).squeeze(2)

    log_prob = (per_token_logps * loss_mask).sum(-1)
    average_log_prob = log_prob / loss_mask.sum(-1)
    # CoT logp
    similarity_scores = torch.matmul(logits, logits.transpose(-1, -2))
    d_k = logits.shape[-1]
    scaled_similarity = similarity_scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32)) 
    
    cross_similarity_scores = compute_cross_similarity_matrix(scaled_similarity, token_types)
    # print(cross_similarity_scores)
    cross_similarity_scores = cross_similarity_scores.unsqueeze(1).expand(-1, CoT_labels.shape[-1], -1)
    
    # type1
    per_token_logps_cot = torch.gather(cross_similarity_scores.log_softmax(-1), dim=2,
                                   index=CoT_labels.unsqueeze(2)).squeeze(2)
    log_prob_cot = per_token_logps_cot.sum(-1)
    average_log_prob_cot = log_prob_cot / CoT_labels.shape[-1]
 

    if return_per_token_logp:
        return per_token_logps

    if return_all:
        return per_token_logps, log_prob, average_log_prob, log_prob_cot, average_log_prob_cot

    return log_prob, average_log_prob, log_prob_cot, average_log_prob_cot


class PreferenceInferenceDataset(torch_data.Dataset):
    def __init__(self,
                 data,
                 tokenizer,
                 image_token_len,
                 img_processor,
                 use_im_start_end=True):

        self.data = data

        self.mm_cfg = {
            'image_processor': img_processor,
            'is_multimodal': True,
            'image_token_len': image_token_len,
            'use_im_start_end': use_im_start_end,
            'keep_image_tag': True
        }
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        sample = self.data[index]
        metainfo = {
            "origin_dataset": sample['origin_dataset'],
            "origin_split": json.loads(sample['origin_split']),
            "origin_idx": sample['idx'],
            "image_id": sample['image_path'],
        }
        question = {'from': 'human', 'value': f"<image>\n{sample['question']}"}
        chosen = {'from': 'gpt', 'value': sample['chosen']}
        rejected = {'from': 'gpt', 'value': sample['rejected']}
        if not sample['image']['bytes']:
            with open(sample['image']['path'], 'rb') as f:
                sample['image']['bytes'] = f.read()
        image = bytes_to_PIL_image(sample['image']['bytes'])

        formated_sample = {
            'image': image,
            "question": question,
            "chosen": chosen,
            "rejected": rejected,
            "idx": sample['idx'],
            "metainfo": metainfo,
            'ch_bbox': sample['ch_bbox'],
            'rej_bbox': sample['rej_bbox']
        }
        preprocess_func= partial(preprocess_v1, has_image=True)
        rej_data_dict, win_data_dict = encode_multimodal_preference_sample(
            formated_sample, self.tokenizer, self.mm_cfg, preprocess_func=preprocess_func)
        return rej_data_dict, win_data_dict

    def __len__(self):
        return len(self.data)


def pretty_print(data_dict, tokenizer):
    input_ids = data_dict['input_ids']
    input_str = tokenizer.decode(input_ids)
    print(f'input_ids.shape={input_ids.shape}\ninput_str is {input_str}')

    label_ids = data_dict['labels']
    print(f'label_ids.shape={input_ids.shape}')
    for i, o in zip(input_ids, label_ids):
        i_tok = tokenizer.convert_ids_to_tokens(i.item())
        o_tok = tokenizer.convert_ids_to_tokens(o.item()) if o.item() != -100 else '[SKIP]'
        print(f'{i_tok:10s} => {o_tok:10s}')


def concate_pad(tensorA, tensorB, padding_value):
    out = torch.nn.utils.rnn.pad_sequence(
        list(tensorA) + list(tensorB),
        batch_first=True,
        padding_value=padding_value)
    return out


import numpy as np


def convert_region_to_token_coords(batch_best_regions, H, W):
    """
    将每个区域的二维坐标转换为一维 token 坐标，支持 tensor 输入，输出 tensor 格式。

    参数:
    - batch_best_regions (Tensor): 形状为 [batch_size, 4] 的 tensor，每行表示一个区域的坐标 (x1, y1, x2, y2)，值在 0 到 1 之间。
    - H (int): 矩阵的高度（行数）。
    - W (int): 矩阵的宽度（列数）。

    返回:
    - batch_best_token_coords (Tensor): 形状为 [batch_size, N] 的 tensor，N 为每个区域一维 token 坐标的数量。
    """
    batch_best_token_coords = []

    # 确保 batch_best_regions 是 tensor
    if isinstance(batch_best_regions, torch.Tensor):
        batch_best_regions = batch_best_regions.detach().cpu().numpy()
    
    for region in batch_best_regions:
        x1, y1, x2, y2 = region
        
        # 将区域坐标转换为实际的二维索引 (假设 x1, x2, y1, y2 都是0到1之间的小数)
        x1, x2 = int(x1 * H), int(x2 * H)  # 对应的行索引
        y1, y2 = int(y1 * W), int(y2 * W)  # 对应的列索引
        
        # 将区域内的每个二维坐标转换为一维 token 坐标
        region_token_coords = []
        for i in range(x1, x2):
            for j in range(y1, y2):
                one_d_token_index = i * W + j
                region_token_coords.append(one_d_token_index)
        
        # 将 region_token_coords 转换为 tensor 并添加到 batch_best_token_coords 中
        batch_best_token_coords.append(torch.tensor(region_token_coords, dtype=torch.long))

    # 将所有的 token 坐标拼接成一个 batch 的 tensor
    batch_best_token_coords = torch.nn.utils.rnn.pad_sequence(batch_best_token_coords, batch_first=True, padding_value=-1)

    return batch_best_token_coords


def preference_collator_fn(instances, pad_token_id):
    rej_instances, win_instances = list(zip(*instances))
    rej_batch = SFT_collator_fn(rej_instances, pad_token_id)
    win_batch = SFT_collator_fn(win_instances, pad_token_id)

    concatenated_input_ids = concate_pad(win_batch['input_ids'], rej_batch['input_ids'], pad_token_id)
    concatenated_labels = concate_pad(win_batch['labels'], rej_batch['labels'], -100)
    concatenated_attention_mask = concatenated_input_ids.ne(pad_token_id)
    win_cot_regions = win_batch['bbox']
    win_cot_label = convert_region_to_token_coords(win_cot_regions, 24, 24)
    rej_cot_regions = rej_batch['bbox']
    rej_cot_label = convert_region_to_token_coords(rej_cot_regions, 24, 24)
    concatenated_cot_labels = torch.cat((win_cot_label, rej_cot_label), dim=0)
    
    batch = dict(
        concatenated_input_ids=concatenated_input_ids,
        concatenated_labels=concatenated_labels,
        concatenated_cot_labels=concatenated_cot_labels,
        concatenated_attention_mask=concatenated_attention_mask,
        win_input_ids=win_batch['input_ids'],
        rej_input_ids=rej_batch['input_ids'],
        win_labels=win_batch['labels'],
        rej_labels=rej_batch['labels'],
        win_labels_cot=win_cot_label,
        rej_labels_cot=rej_cot_label,
        win_attention_mask=win_batch['attention_mask'],
        rej_attention_mask=rej_batch['attention_mask'],
        ch_images=win_batch['images'],
        rej_images=rej_batch['images'],
    )
    return batch




def get_multimodal_sample_logps(model, dataloader, tokenizer, is_llava15=False):
    win_logp_list = []
    rej_logp_list = []

    win_logp_list_cot = []
    rej_logp_list_cot = []

    win_avg_logp_list = []
    rej_avg_logp_list = []

    win_per_token_logp_list = []
    rej_per_token_logp_list = []

    with torch.inference_mode():
        idx=0
        for batch in tqdm.tqdm(dataloader):
            for key in ['win', 'rej']:
                input_ids = batch[f'{key}_input_ids'].cuda()
                # tokens = tokenizer.batch_decode(copy.deepcopy(input_ids))
                # print(tokens)
                labels = batch[f'{key}_labels'].cuda()
                attention_mask = batch[f'{key}_attention_mask'].cuda()
                CoT_labels = batch[f'{key}_labels_cot'].cuda()
                if key == 'win':
                    images = batch['ch_images']
                else:
                    images = batch['rej_images']

                if is_llava15:
                    # print("is llava15")
                    (
                        _,
                        _,
                        _,
                        _,
                        inputs_embeds,
                        labels,
                        token_types
                    ) = model.prepare_cot_inputs_labels_for_multimodal(
                        input_ids=input_ids,
                        position_ids=None,
                        attention_mask=None,
                        past_key_values=None,
                        labels=labels,
                        images=images.to(dtype=torch.bfloat16, device='cuda'),
                    )
                    output = model.forward(
                        inputs_embeds=inputs_embeds,
                        labels=None,
                    )
                else:
                    output = model(
                        input_ids=input_ids,
                        labels=labels,
                        attention_mask=attention_mask,
                        images=images.to(dtype=torch.bfloat16, device='cuda'),
                    )
                per_token_logp, log_prob, average_log_prob, log_prob_cot, average_log_prob_cot = get_batch_logps_cot(output.logits, labels, CoT_labels, token_types, return_all=True)

                # print(per_token_logp.shape, input_ids.shape, labels.shape, flush=True)
                assert per_token_logp.size(1) >= input_ids.size(1) - 1
                per_token_logp = per_token_logp.tolist()
                # per_token_logp = [x[:input_ids[i].ne(tokenizer.pad_token_id).sum().item()] for i, x in enumerate(per_token_logp)]
                log_prob = log_prob.tolist()
                log_prob_cot = log_prob_cot.tolist()
                average_log_prob = average_log_prob.tolist()

                if key == 'win':
                    win_logp_list += log_prob
                    win_avg_logp_list += average_log_prob
                    win_per_token_logp_list += per_token_logp
                    win_logp_list_cot += log_prob_cot
                else:
                    rej_logp_list += log_prob
                    rej_avg_logp_list += average_log_prob
                    rej_per_token_logp_list += per_token_logp
                    rej_logp_list_cot += log_prob_cot
            # print(f'{key} logits in {output.logits.shape}, logp in {log_prob.shape} avg_logp in {average_log_prob.shape}', flush=True)

    return win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list, win_logp_list_cot, rej_logp_list_cot


def write_logp_to_preference_parquet(origin_data, cache_file, logps, overwrite_logps=False):
    out_data = []

    for index in range(len(logps)):
        line = origin_data[index]
        logp_data = {}
        logp_data['logps']=logps[index]

        new_line = copy.deepcopy(line)

        if 'logps' in new_line.keys():
            assert overwrite_logps, 'Found existing logp data, pass overwrite_logps=True to force overwritting'
            new_line['logps'] = json.dumps(logp_data)

        else:
            assert (('question' in list(new_line.keys()))
                    and ('chosen' in list(new_line.keys()))
                    and ('rejected' in list(new_line.keys()))), \
                f'Undefined data structure, expecting [Q, Win, Rej] in keys, got {new_line.keys()}'
            new_line['logps'] = json.dumps(logp_data)

        out_data.append(new_line)

    if torch.distributed.get_rank() == 0:
        step = 5000
        for idx, start in enumerate(range(0, len(out_data), step)):
            temp_data = out_data[start: min(start+step, len(out_data))]
            df = pd.DataFrame(temp_data)
            df.to_parquet(os.path.join(cache_file, f'RLAIF-V-Dataset-withlogp_{idx:03}-{len(temp_data)}.parquet'))

    torch.distributed.barrier()

def inference_logp(model, tokenizer, hf_data, cache_file, image_token_len, img_processor, use_im_start_end, is_llava15=False):
    model = model.to(dtype=torch.bfloat16, device='cuda')
    dataset = PreferenceInferenceDataset(tokenizer=tokenizer,
                                    data = hf_data,
                                    image_token_len=image_token_len,
                                    img_processor=img_processor,
                                    use_im_start_end=use_im_start_end)
    collate_fn = partial(preference_collator_fn, pad_token_id=tokenizer.pad_token_id)
    dataloader = torch_data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn,
                                       num_workers=5, shuffle=False, sampler=InferenceSampler(len(dataset)))

    outputs = get_multimodal_sample_logps(model, dataloader, tokenizer, is_llava15=is_llava15) # win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list

    world_size = torch.distributed.get_world_size()
    merged_outputs = [[None for _ in range(world_size)] for i in range(len(outputs))]
    for i in range(len(outputs)):
        torch.distributed.all_gather_object(merged_outputs[i], outputs[i])
        merged_outputs[i] = [_ for _ in itertools.chain.from_iterable(merged_outputs[i])]


    win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list \
        = merged_outputs
    
    logps = list(zip(win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list))

    write_logp_to_preference_parquet(dataset.data, cache_file, logps, overwrite_logps=False)

    torch.distributed.barrier()

    del model

def inference_logp_cot(model, tokenizer, hf_data, cache_file, image_token_len, img_processor, use_im_start_end, is_llava15=False):
    model = model.to(dtype=torch.bfloat16, device='cuda')
    dataset = PreferenceInferenceDataset(tokenizer=tokenizer,
                                    data = hf_data,
                                    image_token_len=image_token_len,
                                    img_processor=img_processor,
                                    use_im_start_end=use_im_start_end)
    collate_fn = partial(preference_collator_fn, pad_token_id=tokenizer.pad_token_id)
    dataloader = torch_data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn,
                                       num_workers=5, shuffle=False, sampler=InferenceSampler(len(dataset)))

    outputs = get_multimodal_sample_logps(model, dataloader, tokenizer, is_llava15=is_llava15) # win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list

    world_size = torch.distributed.get_world_size()
    merged_outputs = [[None for _ in range(world_size)] for i in range(len(outputs))]
    for i in range(len(outputs)):
        torch.distributed.all_gather_object(merged_outputs[i], outputs[i])
        merged_outputs[i] = [_ for _ in itertools.chain.from_iterable(merged_outputs[i])]


    win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list, win_logp_list_cot, rej_logp_list_cot \
        = merged_outputs

    logps = list(zip(win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list, win_logp_list_cot, rej_logp_list_cot))

    write_logp_to_preference_parquet(dataset.data, cache_file, logps, overwrite_logps=False)

    torch.distributed.barrier()

    del model