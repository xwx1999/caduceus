import torch


def mlm_getitem(seq, mlm_probability=0.15, contains_eos=False, tokenizer=None, eligible_replacements=None):
    """Helper method for creating MLM input / target.

    Adapted from:
    https://github.com/huggingface/transformers/blob/14666775a296a76c88e1aa686a9547f393d322e2/src/transformers/data/data_collator.py#L751

    首先，根据 contains_eos 的值决定是否从序列中移除结束符。
    创建一个与目标张量形状相同的 probability_matrix，填充 mlm_probability 值。
    使用 torch.bernoulli 函数根据概率矩阵生成一个随机的掩盖索引矩阵 masked_indices。
    将目标张量中未被掩盖的索引位置设置为填充 token 的 ID (tokenizer.pad_token_id)，因为只有这些位置会计算损失。

    80% 的时间，将掩盖的输入 token 替换为特殊的掩码 token (tokenizer.mask_token)。
    10% 的时间，将掩盖的输入 token 替换为随机的词（从 eligible_replacements 或 tokenizer 的词汇表中随机选择）。
    剩余的 10% 时间，保持掩盖的输入 token 不变。

    """
    data = seq[:-1].clone() if contains_eos else seq.clone()  # remove eos, if applicable
    target = data.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(target.shape, mlm_probability)
    # TODO: Do we need to avoid "masking" special tokens as is done here?
    #  https://github.com/huggingface/transformers/blob/14666775a296a76c88e1aa686a9547f393d322e2/src/transformers/data/data_collator.py#L760-L766
    masked_indices = torch.bernoulli(probability_matrix).bool()
    target[~masked_indices] = tokenizer.pad_token_id  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(target.shape, 0.8)).bool() & masked_indices
    data[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(target.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    if eligible_replacements is not None:
        rand_choice = torch.randint(eligible_replacements.shape[0], size=target.shape)
        random_words = eligible_replacements[rand_choice]
    else:
        random_words = torch.randint(len(tokenizer), size=target.shape, dtype=torch.long)
    data[indices_random] = random_words[indices_random]
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return data, target
