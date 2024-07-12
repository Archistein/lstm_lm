import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from math import exp


class JokesDataset(Dataset):

    def __init__(self, jokes):
        self.jokes = jokes

    def __len__(self) -> int:
        return len(self.jokes)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.jokes[idx]
        y = torch.cat((x[1:], torch.zeros(1, dtype=torch.long)))

        return x, y


def get_data_loaders(data: torch.Tensor, split_ratio: float = 0.9, batch_size: int = 64) -> tuple[DataLoader, DataLoader]:
    train_size = int(len(data) * split_ratio)
    test_size = len(data) - train_size

    dataset = JokesDataset(data)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_data_loader, test_data_loader


@torch.inference_mode
def evaluate(model: nn.Module, criterion: callable, data_loader: DataLoader, device: torch.device) -> float:
    model.eval()

    average_loss = 0

    for seqs, targets in data_loader:
        h, c = model.init_hidden(len(seqs), device)

        seqs, targets = seqs.to(device), targets.to(device)

        logits, (h, c) = model(seqs, (h, c))

        average_loss += criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

    model.train()

    return average_loss.item()/len(data_loader)


def trainer(model: nn.Module, 
          data: torch.Tensor,
          device: torch.device,
          batch_size: int = 64,
          epoch: int = 10, 
          lr: float = 3e-4, 
          padding_idx: int = 0, 
          grad_clip: int = 5,) -> None:

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)

    train_data_loader, test_data_loader = get_data_loaders(data, 0.9, batch_size)

    model.train()
    model.to(device)

    for e in range(epoch):

        for i, (seqs, targets) in enumerate(train_data_loader):

            h, c = model.init_hidden(len(seqs), device)

            seqs, targets = seqs.to(device), targets.to(device)

            optimizer.zero_grad()

            logits, (h, c) = model(seqs, (h, c))
            
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            if (i % (len(train_data_loader)//5) == 0):
                val_loss = evaluate(model, criterion, test_data_loader, device)
                print(f'Val_Loss: {val_loss:.4f} Val_Perplexity: {exp(val_loss)}')

            if (i % (len(train_data_loader)//50) == 0):
                print(f'Epoch [{e}/{epoch-1}] Batch [{i}/{len(train_data_loader)-1}] Loss: {loss:.4f} Perplexity: {exp(loss):.4f}')
    