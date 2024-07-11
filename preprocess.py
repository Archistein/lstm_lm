import torch
import pandas as pd
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


def data_load(path: str, col: str) -> pd.Series:
    data = pd.read_csv(path, usecols=[col])

    data[col] = data[col].str.lower().replace(r'[\x08\x10]', '', regex=True) 

    max_length = int(data[col].str.len().quantile(0.95))
    data = data.loc[data[col].str.len() <= max_length].reset_index()

    return data[col]


def get_bpe_tokenizer(data: pd.Series, special_tokens: list[str], vocab_size: int = 512, min_frequency: int = 2) -> Tokenizer:

    tokenizer = Tokenizer(models.BPE())

    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace('⠀')

    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=special_tokens)
    tokenizer.train_from_iterator(data, trainer)

    tokenizer.decoder = decoders.BPEDecoder()
    
    return tokenizer


def byte_pair_encode(data: pd.Series, tokenizer: Tokenizer, sos_tok: int = 1, eos_tok: int = 2) -> torch.Tensor:
    max_length = data.str.len().max()
    data_bpe = torch.zeros(len(data), max_length+2, dtype=torch.long)

    for i in range(len(data)):
        toks = torch.tensor(tokenizer.encode(data[i]).ids, dtype=torch.long)

        data_bpe[i][0] = sos_tok
        data_bpe[i][1:1+len(toks)] = toks
        data_bpe[i][1+len(toks)] = eos_tok

    return data_bpe