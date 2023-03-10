# AI shit

yes this repo is titled reddit-gpt, and that's because i was originally going to train my own model on reddit comments, but i totally underestimated how much computing power i need (i only have an RTX 3070 8GB).

instead, im using this repo for random small AI stuff.

I use an unconventional (some may say nonexistent) naming / file heirarchy scheme, but the general rule is "\*gpu\*" files are for training. "\*data\*" files interact with whatever data is in `./resources`. "\*test\*" or "\*interactive\*" files are for testing the model.

## Requirements
*note, you may want to use a venv (python -m venv .venv), but idk its whatever man*
```bash
pip install -r requirements.txt
```
