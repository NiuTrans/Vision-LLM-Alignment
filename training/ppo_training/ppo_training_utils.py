import torch
import torch.nn.functional as F


def sampling(actor_model,
            img, lang, 
            attention_mask=None,
            input_labels=None,
            max_seq_len=1024,
            pad_token_id=0):
    
    generation_kwargs={
        "top_k": 50,
        "top_p": 0.95,
        "do_sample": True,
        "max_new_tokens": 256,
        "num_return_sequences": 1
    }

    max_new_tokens = generation_kwargs["max_new_tokens"]
    generation_kwargs.pop("max_new_tokens")

    batch_size = lang.size()[0]

    all_res = []
 
    actor_model.eval()

    for index in range(batch_size):
        try:
            sub_img = img[index].unsqueeze(0)
        except:
            sub_img = [img[index]]   # reused by the prediction, and there is a situation where the image is None.
        
        sub_attention_mask = attention_mask[index]
        
        sub_lang = lang[index].unsqueeze(0)[sum(sub_attention_mask==pad_token_id):]

        res = actor_model.generate(sub_img, sub_lang, 
                                        generation_length=max_new_tokens+max_seq_len, 
                                        generation_kwargs=generation_kwargs)
        
        all_res.append(res)
    actor_model.train()
    return all_res

def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)

def compute_logprobs_from_actor_and_ref(actor_model,
                                    ref_model,
                                    images,
                                    input_ids,
                                    input_labels=None,
                                    attention_mask=None,
                                    image_num=None):
    with torch.no_grad():
        logits = actor_model(
                    images,
                    input_ids,
                    attention_mask=attention_mask,
                    input_labels=input_labels,
                    image_num=image_num)[1]

        ref_logits = ref_model(
                    images,
                    input_ids,
                    attention_mask=attention_mask,
                    input_labels=input_labels,
                    image_num=image_num)[1]
    
    logprobs = gather_log_probs(logits[:, :-1, :], input_ids[:, 1:])
    ref_logprobs = gather_log_probs(ref_logits[:, :-1, :], input_ids[:,1:])
    
    return logprobs, ref_logprobs

def compute_kl_reward_scores(logprobs, ref_logprobs, reward_scores,
                            start, attention_mask):
    
    # some hyper-parameters from computing rewards
    # ref: https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/dschat/rlhf/ppo_trainer.py#L66
    kl_ctl = 0.1
    clip_reward_value = 5

    kl_divergence_estimate = - kl_ctl * (logprobs - ref_logprobs)
    kl_rewards = kl_divergence_estimate
    kl_distance = kl_divergence_estimate
    ends = start + attention_mask[:, start:].sum(1) + 1

    reward_clip = torch.clamp(reward_scores, -clip_reward_value,
                            clip_reward_value)

    batch_size = logprobs.shape[0]

    for j in range(batch_size):
        kl_rewards[j, start:ends[j]][-1] += reward_clip[j]
    return kl_rewards, torch.abs(kl_distance).mean()

def get_advantages_and_returns(values, rewards, start):
    # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
    gamma = 0.99
    lam = 0.95
    lastgaelam = 0
    
    advantages_reversed = []
    length = rewards.size()[-1]
    for t in reversed(range(start, length)):
        nextvalues = values[:, t + 1] if t < length - 1 else 0.0
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values[:, start:]
    return advantages.detach(), returns

def critic_loss_fn(values, old_values, returns, mask):
    cliprange_value = 0.2

    values_clipped = torch.clamp(
        values,
        old_values - cliprange_value,
        old_values + cliprange_value,
    )
    vf_loss1 = (values - returns)**2
    vf_loss2 = (values_clipped - returns)**2
    vf_loss = 0.5 * torch.sum(
        torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
    return vf_loss

def actor_loss_fn(logprobs, old_logprobs, advantages, mask):
    ## policy gradient loss
    cliprange = 0.2
    log_ratio = (logprobs - old_logprobs) * mask
    ratio = torch.exp(log_ratio)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - cliprange,
                                            1.0 + cliprange)
    pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
    return pg_loss