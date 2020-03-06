from pytorch_pretrained_bert import BertTokenizer
import numpy as np
from utils import write_json
x=[1,2,3,4,5,6]
write_json('./data/x.json', x)