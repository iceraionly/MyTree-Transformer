import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import pdb
import pickle
import os
import json
import ast
import re
from model import subsequent_mask
from pytorch_pretrained_bert import BertTokenizer

#定义一个训练时需要的批次数据对象，封装了用于训练的src和tgt句子，以及mask
class Batch(object):

    def __init__(self, src, trg=None, pad=0):
        self.src = src # B 个序列[1,5,3, 0]
        self.src_mask = (src != pad).unsqueeze(-2)  # [[1,1,1,0]]
        if trg is not None:
            self.trg = trg[:, :-1]  #
            self.trg_y = trg[:, 1:]  # 后挪一个位置开始
            self.trg_mask = self.make_std_mask(self.trg, pad) # 训练时的有效预测个数
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "创建Mask，使得我们不能attend to未来的词"
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


# 定义一个训练函数用于训练和计算损失、更新梯度
def run_epoch(data_iter, model, loss_compute, device):
    """提供训练和日志功能"""

    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):

        src = batch.src.to(device)
        trg = batch.trg.to(device)
        src_mask = batch.src_mask.to(device)
        trg_mask = batch.trg_mask.to(device)
        trg_y = batch.trg_y.to(device)
        ntokens = batch.ntokens.to(device)
        # print(src)
        out = model.forward(src, trg, src_mask, trg_mask)
        loss = loss_compute(out, trg_y, ntokens)

        total_loss += loss.detach().cpu().numpy()
        total_tokens += ntokens.cpu().numpy()
        tokens += ntokens.cpu().numpy()
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss.detach().cpu().numpy() / ntokens.cpu().numpy(), tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

##优化函数
class NoamOpt(object):
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 1, 50,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    # 50 预热步数内线性增加学习速率 之后减小

##正则化
class LabelSmoothing(nn.Module):
    "实现labelsmoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

def make_save_dir(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir
def read_json(filename):
    with open(filename, 'r') as fp:
        data = json.load(fp)
    return data
def write_json(filename,data):
    with open(filename, 'w') as fp:
        json.dump(data, fp)

##数据工具
class data_utils():
    def __init__(self, args):
        self.seq_length = args.seq_length
        self.batch_size = args.batch_size

        self.dict_path = os.path.join(args.model_dir, 'dictionary.json')
        self.train_path = args.train_path
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.eos_id = 0
        self.unk_id = 1
        self.mask_id = 2
        self.cls_id = 3

        if args.train or not os.path.exists(self.dict_path):
            self.process_training_data()
        elif args.test:
            self.new_vocab = read_json(self.dict_path)

        print('vocab_size:', len(self.new_vocab))

        self.vocab_size = len(self.new_vocab)
        self.index2word = self.vocab_size * [[]]
        for w in self.new_vocab:
            self.index2word[self.new_vocab[w]] = w

    def process_training_data(self):
        self.training_data_code = []
        self.training_data_summary = []
        with open(self.train_path, "r") as f:
            all = [line.strip() for line in f.readlines()]
        all_dict = []
        for i in all:
            all_dict.append(ast.literal_eval(i))

        self.new_vocab = dict()
        self.new_vocab['[PAD]'] = 0
        self.new_vocab['[UNK]'] = 1
        self.new_vocab['[MASK]'] = 2
        self.new_vocab['[CLS]'] = 3

        co = []
        su = []
        word_count = {}
        for line in all_dict:
            co_list = []
            su_list = []
            pattern = "[A-Z.]"


            ll = line.get("nl").strip()
            ll = re.sub(pattern, lambda x: " " + x.group(0), ll) #解决峰驼
            ll = re.sub(";", " ; ", ll) #保留；
            for word in re.split(" |\(|\)|\{|\}|,|\n|\[|\]|\.",ll):
                w=""
                is_skip=False
                for sub_words in self.tokenizer.tokenize(word):
                    if "##" in sub_words:
                        is_skip = True
                        break
                    if w.strip()=="":
                        w=sub_words
                if is_skip:
                    w='[UNK]'
                    su_list.append(w)
                else:
                    if w.strip()!="":
                        word_count[w] = word_count.get(w, 0) + 1
                        su_list.append(w)
            su_list = ['[CLS]'] + su_list
            su.append(su_list)

            ll = line.get("code").strip()
            ll = re.sub(pattern, lambda x: " " + x.group(0), ll)  # 解决峰驼
            ll = re.sub(";", " ; ", ll)  # 保留；
            for word in re.split(" |\(|\)|\{|\}|,|\n|\[|\]|\.",ll):
                w = ""
                is_skip = False
                for sub_words in self.tokenizer.tokenize(word):
                    if "##" in sub_words:
                        is_skip = True
                        break
                    if w.strip() == "":
                        w = sub_words
                if is_skip:
                    w = '[UNK]'
                    co_list.append(w)
                else:

                    if w.strip() != "":

                        word_count[w] = word_count.get(w, 0) + 1
                        co_list.append(w)
            co_list = ['[CLS]'] + co_list
            co.append(co_list)

        for w in word_count:
            if word_count[w] >= 1:
                self.new_vocab[w] = len(self.new_vocab)

        for d in co:
            word_list = []
            for w in d:
                if w in self.new_vocab:
                    word_list.append(self.new_vocab[w])
                else:
                    word_list.append(self.unk_id)
            if( len(word_list)<self.seq_length ):
                word_list = list(word_list + [0] * (self.seq_length - len(word_list)))
            else:
                word_list=word_list[:self.seq_length]
            # print(word_list)
            self.training_data_code.append(word_list)

        for d in su:
            word_list = []
            for w in d:
                if w in self.new_vocab:
                    word_list.append(self.new_vocab[w])
                else:
                    word_list.append(self.unk_id)
            if (len(word_list) < self.seq_length):
                word_list = list(word_list + [0] * (self.seq_length - len(word_list)))
            else:
                word_list = word_list[:self.seq_length]
            self.training_data_summary.append(word_list)

        write_json(self.dict_path, self.new_vocab)

    def data_load(self,input_data,output_data,batch_size):
        data_size = len(input_data)
        num_batches= int((data_size  - 1) / batch_size) + 1


        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            src = Variable(torch.from_numpy(self.boolean_indexing(input_data[start_index:end_index])).long(), requires_grad=False)
            tgt = Variable(torch.from_numpy(self.boolean_indexing(output_data[start_index:end_index])).long(), requires_grad=False)
            yield Batch(src, tgt, 0)

    def boolean_indexing(self,v):
        lens = np.array([len(item) for item in v])
        mask = lens[:, None] > np.arange(lens.max())
        out = np.zeros(mask.shape, dtype=int)
        out[mask] = np.concatenate(v)
        return out

    def text2id(self, text):
        vec = np.zeros([self.seq_length] ,dtype=np.int32)
        unknown = 0.
        pattern = "[A-Z.]"
        w_list = []

        ll = text.strip()
        ll = re.sub(pattern, lambda x: " " + x.group(0), ll)  # 解决峰驼
        ll = re.sub(";", " ; ", ll)  # 保留；
        for word in re.split(" |\(|\)|\{|\}|,|\n|\[|\]|\.", ll):

            w = ""
            is_skip = False
            for sub_words in self.tokenizer.tokenize(word):
                if "##" in sub_words:
                    is_skip = True
                    break
                if w.strip() == "":
                    w = sub_words
            if is_skip:
                w = '[UNK]'
            if w in self.new_vocab:
                w_list.append(self.new_vocab[w])
            else:
                w_list.append(self.unk_id)

        w_list = [self.new_vocab['[CLS]']] + w_list
        indexed_tokens = w_list
        # assert len(text.strip().split())+1 == len(indexed_tokens)

        for i,word in enumerate(indexed_tokens):
            if i >= self.seq_length:
                break
            vec[i] = word
        return vec

    def id2sent(self,indices, test=False):
        sent = []
        word_dict={}
        for w in indices:
            if w != self.eos_id:
                sent.append(self.index2word[w])
                # print(sent)
        return ' '.join(sent)

##简单loss计算
class SimpleLossCompute(object):
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss * norm.float()
        # return (loss * norm).item()

#贪心解码
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys