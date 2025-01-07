from script import Run

# THIS IS MAIN SCRIPT.

EMBEDDING_PATH="../PLAYGROUND/features/lfw_embeddings.pt"
BIN_PATH="../PLAYGROUND/features/lfw.bin"
TITLE="hellu"
EXPAND_DIM=1024
NONZERO=14
"""
CHOOSE EXPAND_DIM:int [512,1024,2048,4096,8192]
RANGE OF NONZERO:int 10<= nonzero <= 16

"""

Run(_embedding_path=EMBEDDING_PATH,
    _bin_path=BIN_PATH,
    _title=TITLE,
    _expand_dim=EXPAND_DIM,
    _nonzero=NONZERO
   )