from model import *
from utils import *
import os
import subprocess
import random
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

class Solver():
    def __init__(self, args):
        self.args = args
        self.model_dir = make_save_dir(args.model_dir)
        if not os.path.exists(os.path.join(self.model_dir, 'code')):
            os.makedirs(os.path.join(self.model_dir, 'code'))

        self.data_utils = data_utils(args)
        V=len(self.data_utils.new_vocab) #字典长度
        # V=2000
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.1) #size为src_vocab
        self.model = make_model(V,V,6)
        self.model = self.model.to(self.device)



    def train(self):
        if self.args.load:
            path = os.path.join(self.model_dir, 'model.pth')
            self.model.load_state_dict(torch.load(path)['state_dict'])
        tt = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # print(name)
                ttt = 1
                for s in param.data.size():
                    ttt *= s
                tt += ttt
        print('total_param_num:', tt)
        model_opt = get_std_opt(self.model)

        # data_yielder = self.data_utils.train_data_yielder()
        # optim = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
        # optim = BertAdam(self.model.parameters(), lr=1e-4)
        with open(self.args.test_path, "r") as f:
            all = [line.strip() for line in f.readlines()]
        all_dict = []
        for i in all:
            all_dict.append(ast.literal_eval(i))
        src_input=[]
        src_output=[]
        for x in all_dict:
            src_input.append(self.data_utils.text2id(x.get("code").strip()))
            src_output.append(self.data_utils.text2id(x.get("nl").strip()))
        train_rec=[]
        test_rec=[]
        for epoch in range(self.args.num_step):
            print("Epoch %d :" %(epoch))
            self.model.train()
            tr = run_epoch(self.data_utils.data_load(self.data_utils.training_data_code,self.data_utils.training_data_summary,self.args.batch_size), self.model,
                      SimpleLossCompute(self.model.generator, self.criterion, model_opt), self.device)
            self.model.eval()
            te = run_epoch(self.data_utils.data_load(src_input, src_output, self.args.batch_size), self.model,
                            SimpleLossCompute(self.model.generator, self.criterion, None), self.device)
            train_rec.append(tr)
            test_rec.append(te)

        write_json('./data/train_recorde.json', train_rec)
        write_json('./data/test_recorde.json', test_rec)


        print('saving!!!!')
        model_name = 'model.pth'
        state = {'step': 0, 'state_dict': self.model.state_dict()}
        torch.save(state, os.path.join(self.model_dir, model_name))

    def test(self, threshold=0.8):
        path = os.path.join(self.model_dir, 'model.pth')
        self.model.load_state_dict(torch.load(path)['state_dict'])

        with open(self.args.test_path, "r") as f:
            all = [line.strip() for line in f.readlines()]
        all_dict = []
        for i in all:
            all_dict.append(ast.literal_eval(i))
        print("ok")

        # src_input=[]
        # src_output=[]
        # for x in all_dict:
        #     src_input.append(self.data_utils.text2id(x.get("code").strip()))
        #     src_output.append(self.data_utils.text2id(x.get("nl").strip()))
        #
        # for epoch in range(1):
        #     print("Epoch %d :" % (epoch))
        #     self.model.eval()
        #     print(run_epoch(self.data_utils.data_load(src_input,src_output,self.args.batch_size), self.model,
        #                 SimpleLossCompute(self.model.generator, self.criterion, None), self.device))

        self.model.eval()
        score_sum=0.
        for x in all_dict:
            # print(all_dict[23].get("code").strip())
            src = Variable(torch.from_numpy(np.expand_dims(self.data_utils.text2id(x.get("code").strip()),axis=0)).long())
            # print(src)

            # print(self.data_utils.id2sent(self.data_utils.text2id(all_dict[23].get("code").strip())))
            src_mask = Variable(torch.ones(1, 1, 60)) #一个 一行60列的矩阵
            res = greedy_decode(self.model, src, src_mask, max_len=60, start_symbol=1)#max_len为输出的语句长度
            # print(res)
            candidate = self.data_utils.id2sent(np.squeeze(res.cpu().numpy())).split(' ')
            seq = x.get("nl").strip().split(' ')
            reference = []
            reference.append(seq)
            smooth = SmoothingFunction()  # 定义平滑函数对象
            score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25),smoothing_function=smooth.method2)
            print(score)
            score_sum+=score
        print(score_sum/len(all_dict))