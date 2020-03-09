from pytorch_pretrained_bert import BertTokenizer
import numpy as np
from utils import write_json

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(s):
    w_list = ["<s>"]
    pattern = "[A-Z.]"
    s = re.sub(pattern, lambda x: " " + x.group(0), s)
    print(s)
    words = s.split()
    print(words)
    for word in words:
        tokens = []
        is_skip = False
        for token_word in tokenizer.tokenize(word):
            if "##" in token_word:
                is_skip = True
                break
            tokens.append(token_word)
        if is_skip:
            w_list.append('<UNK>')
        else:
            w_list.extend(tokens)
    w_list.append("</s>")
    return w_list

tokenize("ABC.DEF123JKL****abc")