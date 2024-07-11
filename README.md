# Language model for generating jokes

This repo contains a long short-term memory RNN based language model for generating jokes. 

> [!Warning]
> Since the dataset was taken from [here](https://www.kaggle.com/datasets/abhinavmoudgil95/short-jokes), I cannot be held responsible for the presence of obscenity.

## Demo     
![](demo.gif)

## Model

```
LSTM(
  (embed): Embedding(512, 64, padding_idx=0)
  (lstm): LSTM(64, 512, num_layers=2, batch_first=True)
  (fc): Linear(in_features=512, out_features=512, bias=True)
)
```

## Usage

To retrain model just change mode in `main.py` to 1 (training mode).

To generate:

```bash
$ python main.py
Prompt: q:
```

Type in your prompt (or don't) and press enter:

```bash
$ python main.py
Prompt: q: what is the difference between a bad bird and a fox? a: they are both stuck up cunts.
```

## Requirements

+ pandas==2.2.2
+ tokenizers==0.19.1
+ torch==2.3.1

### Have a fun :)