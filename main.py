import os
from preprocess import *
from tokenizers import Tokenizer
from model import LSTM, sample
from train import trainer

# Operating mode
# 0 - Inference mode
# 1 - Training mode
MODE = 0

def main() -> None:
    
    data_root = 'data'
    tokenizer_root = 'tokenizer'
    weights_root = 'weights'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    vocab_size = 512
    n_emb = 64
    n_hidden = 512
    n_layers = 2

    model = LSTM(vocab_size, n_emb, n_hidden, n_layers)
    model.to(device)

    if MODE:
        print('Training mode enabled')
        print(f'Detected device: {device.type}')

        batch_size = 64
        epoch = 5
        lr = 3e-4

        special_tokens = ['<pad>', '<sos>', '<eos>']

        print('Start data processing')
        data = data_load(f'{data_root}/jokes.csv', 'joke')
        tokenizer = get_bpe_tokenizer(data, special_tokens, vocab_size)
        data = byte_pair_encode(data, tokenizer)
        
        if not os.path.exists(tokenizer_root): os.mkdir(tokenizer_root)
        tokenizer.save(f'{tokenizer_root}/tokenizer.json')

        print('Start training the model')
        trainer(model, data, device, batch_size, epoch, lr)
        # torch.save(model.state_dict(), f'{weights_root}/params.pt')
    else:
        tokenizer = Tokenizer.from_file(f'{tokenizer_root}/tokenizer.json')
        model.load_state_dict(torch.load(f'{weights_root}/params.pt'))

    temperature = 0.6

    while True:
        try:
            prompt = input('Prompt: ')
        except EOFError as e:
            break

        print(f'\033[FPrompt: {prompt}', end='')
        for tok in sample(model, tokenizer, device, prompt, temperature):
            print(tok, end='', flush=True)
        print()


if __name__ == '__main__':
    main()