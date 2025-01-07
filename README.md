# [Deep Face Template Protection in the Wild](https://www.sciencedirect.com/science/article/abs/pii/S0031320324010872)

> Experiment source code for public use

## Getting Started

#### Install Dependencies

Install all libraries used in this project.

#### Setup Variables

1. open main.py
2. Set variables : EMBEDDING_PATH, BIN_PATH,TITLE, EXPAND_DIM, NONZERO
   > Note that you should choose `EXPAND_DIM (int)` among 5 : `[512,1024,2048,4096,8192]`, and the range of `NONZERO (int)` is from 10 to 16.

#### Start Experiment

After setting up variables, start experiment locally.

```
python main.py
```

Progress will be displayed on the terminal.
