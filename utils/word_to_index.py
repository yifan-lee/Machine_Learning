def tokenize(texts, vocab_size=5000, max_len=100):
    # 1. 全部转小写并拆分
    token_lists = [t.lower().split() for t in texts]

    # 2. 统计词频
    word_freq = {}
    for tokens in token_lists:
        for tok in tokens:
            word_freq[tok] = word_freq.get(tok, 0) + 1

    # 3. 选出高频词
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    vocab_words = [w for w, _ in sorted_words[:vocab_size-2]]

    word2idx = {w: i+2 for i, w in enumerate(vocab_words)}
    word2idx["<PAD>"] = 0
    word2idx["<UNK>"] = 1

    # 4. 转为索引并补齐
    encoded = []
    for tokens in token_lists:
        ids = [word2idx.get(tok, 1) for tok in tokens]
        ids = ids[:max_len] + [0] * (max_len - len(ids))
        encoded.append(ids)
    return encoded, word2idx



