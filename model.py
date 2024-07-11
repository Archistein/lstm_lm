import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from typing import Generator


class LSTM(nn.Module):

    def __init__(self, vocab_size: int, n_emb: int, n_hidden: int, n_layers: int, padding_idx: int = 0):
        super().__init__()

        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab_size, n_emb, padding_idx=padding_idx)
        self.lstm = nn.LSTM(n_emb, n_hidden, n_layers, batch_first=True)

        self.fc = nn.Linear(n_hidden, vocab_size)

    def forward(self,
                x: torch.Tensor,
                hc: tuple[torch.Tensor, torch.Tensor]
               ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:

        emb = self.embed(x)

        out, (h, c) = self.lstm(emb, hc)

        logits = self.fc(out)

        return logits, (h, c)

    def init_hidden(self, batch_size: int, device: torch.device):
        return (torch.zeros(self.n_layers, batch_size, self.n_hidden, device=device),
                torch.zeros(self.n_layers, batch_size, self.n_hidden, device=device))


@torch.inference_mode
def sample(model: nn.Module, 
           tokenizer: Tokenizer, 
           device: torch.device, 
           prompt: str = '',
           temperature: int = 0.5,
           max_length: int = 200, 
           sos_tok: int = 1,
           eos_tok: int = 2
          ) -> Generator[str, None, None]:

    seq = [sos_tok] + tokenizer.encode(prompt.lower()).ids

    for _ in range(max_length):
        t = torch.tensor(seq, device=device).unsqueeze(0)

        logits = model.forward(t, model.init_hidden(1, device))[0][0][-1]

        next_tok = torch.multinomial(F.softmax(logits / temperature, dim=0), 1).item()
        seq.append(next_tok)

        if next_tok == eos_tok:
            break

        yield tokenizer.decode([next_tok])