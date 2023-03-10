
from json import load
import bpe

with open("resources/messages.json") as f:
    messages:list[str] = [i.strip("\n") for i in load(f) if i]

bpe_encoder = bpe.Encoder(vocab_size=1440) # 1440 is a good number because it is a monitor resolution :p

bpe_encoder.fit(messages)

example = "this is big test poggers"
tokens = bpe_encoder.tokenize(example)
print(tokens)
transformed = bpe_encoder.transform([example])
print(next(transformed))
print(next(bpe_encoder.inverse_transform(bpe_encoder.transform([example]))))
