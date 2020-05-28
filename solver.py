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
        self.criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.1).cuda() #size为tag_vocab

        self.model = make_model(V,V,6)
        self.model = self.model.to(self.device)



    def train(self):
        if self.args.load:
            path = os.path.join(self.model_dir, 'model.pth')
            self.model.load_state_dict(torch.load(path)['state_dict'])

        tt = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                ttt = 1
                for s in param.data.size():
                    ttt *= s
                tt += ttt
        print('total_param_num:', tt)

        model_opt = NoamOpt(self.model.src_embed[0].d_model, 1, 2000,
                   torch.optim.Adam(self.model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

        # optim = BertAdam(self.model.parameters(), lr=1e-4)
        train_rec=[]
        for epoch in range(self.args.num_step):
            self.model.train()
            start = time.time()
            tr = run_epoch(self.data_utils.data_load(self.data_utils.training_data_code,self.data_utils.training_data_summary,self.args.batch_size), self.model,
                      SimpleLossCompute(self.model.generator, self.criterion, model_opt), self.device)
            elapsed = time.time() - start
            print('----------epoch: %d end, total loss= %f , train_time= %f Sec -------------' % (epoch, tr, elapsed))
            train_rec.append(tr)

            print('saving!!!!')
            model_name = 'model.pth'
            state = {'step': epoch, 'state_dict': self.model.state_dict()}
            torch.save(state, os.path.join(self.model_dir, model_name))

        print('training process end, total_loss is =', train_rec)
        write_json('./data/train_recorde.json', train_rec)


    def test(self, threshold=0.8):
        path = os.path.join(self.model_dir, 'model.pth')
        self.model.load_state_dict(torch.load(path,map_location=lambda storage, loc: storage)['state_dict'])

        with open(self.args.test_path, "r") as f:
            all = [line.strip() for line in f.readlines()]
        all_dict = []
        for i in all:
            all_dict.append(ast.literal_eval(i))
        print("ok")

        # self.model.eval()
        # x=all_dict[25]
        # print(x.get("code").strip())
        # src = Variable(torch.from_numpy(np.expand_dims(self.data_utils.text2id(x.get("code").strip()),
        #                                                    axis=0)).long()).cuda()
        # print(src)
        # print(self.data_utils.id2sent(self.data_utils.text2id(x.get("code").strip())))
        # src_mask = Variable(torch.ones(1, 1, self.args.seq_length)).cuda() # 一个 一行60列的矩阵
        # res = greedy_decode(self.model, src, src_mask, max_len=self.args.seq_length,
        #                     start_symbol=3)#max_len为输出的语句长度
        # print(res)
        # res_text=self.data_utils.id2sent(np.squeeze(res.cpu().numpy()))
        # print(res_text)
        # print(x.get("nl").strip())


        self.model.eval()
        score_sum=0.
        num=0
        for x in all_dict:


            src = Variable(torch.from_numpy(np.expand_dims(self.data_utils.text2id(x.get("code").strip()),
                                                           axis=0)).long()).cuda()
            src_mask = Variable(torch.ones(1, 1, self.args.seq_length)).cuda()#一个 一行60列的矩阵
            res = greedy_decode(self.model, src, src_mask, max_len=self.args.seq_length, start_symbol=3)#max_len为输出的语句长度


            candidate = self.data_utils.id2sent(np.squeeze(res.cpu().numpy())).split(' ')
            seq = x.get("nl").strip().split(' ')
            reference = []
            reference.append(seq)
            smooth = SmoothingFunction()  # 定义平滑函数对象
            score = sentence_bleu(reference, candidate, weights=(0, 1, 0, 0),smoothing_function=smooth.method2)
            num+=1
            score_sum+=score
            print(score_sum/num)
        print("mean:",score_sum/len(all_dict))
