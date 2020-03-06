import matplotlib.pyplot as plt
from model import *
from utils import *

class test_T():

    def test1():
        # 测试MultiHeadedAttention的过程
        batch_size=16
        L = 50         # 序列长度
        d_model = 512  # 词向量维度
        h = 8
        x = torch.randn(batch_size, L, d_model)  # 生层一个测试序列x
        print(x.size())

    def test2():
        # 测试MultiHeadedAttention的过程
        obj = MultiHeadedAttention(8, 512)
        q = torch.randn(2, 10, 512)  # 序列输入x
        line_net = clones(nn.Linear(512, 512), 4)

        q, k, v = [l(x).view(2, -1, 8, 64).transpose(1, 2) for l, x in zip(line_net, (q, q, q))]
        print(k.size(), k.transpose(-2, -1).size())
        d_model=512
        h=8
        d_k = d_model // h
        print("d_k:", d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        print("soc:", scores.size())
        attn = F.softmax(scores, dim=-1)
        print("attn size: ", attn.size())
        r_x = torch.matmul(attn, v)
        print(r_x.size())

        out = r_x.transpose(1, 2).contiguous().view(2, -1, 8 * 64)
        print(out.size())  # [2, 10, 512]

    def test3():
        # 在位置编码下方，将基于位置添加正弦波。对于每个维度，波的频率和偏移都不同。
        plt.figure(figsize=(15, 5))
        pe = PositionalEncoding(20, 0)
        y = pe.forward(Variable(torch.zeros(1, 100, 20)))
        plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
        plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
        plt.show()

    def test4():
        #建模
        tmp_model = make_model(10,10,2)
        print(tmp_model)

    def test5():
        # 不同大小与不同超参模型的学习率曲线
        opts = [NoamOpt(512, 1, 4000, None),
                NoamOpt(512, 1, 8000, None),
                NoamOpt(256, 1, 4000, None)]
        plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
        plt.legend(["512:4000", "512:8000", "256:4000"])
        plt.show()

    def test6():
        # Example of label smoothing.
        crit = LabelSmoothing(5, 0, 0.4)
        predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                                     [0, 0.2, 0.7, 0.1, 0],
                                     [0, 0.2, 0.7, 0.1, 0]])
        v = crit(Variable(predict.log()),Variable(torch.LongTensor([2, 1, 0])))
        plt.imshow(crit.true_dist)
        plt.show()

    def test7():
        crit = LabelSmoothing(5, 0, 0.1)
        def loss(x):
            d = x + 3 * 1
            predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d], ])
            # print(predict)
            return crit(Variable(predict.log()), Variable(torch.LongTensor([1]))).item()
        plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])
        plt.show()



if __name__ == '__main__':

    V = 11
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.1)
    model = make_model(V, V, N=2)
    model = model.to(device)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 4000,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    for epoch in range(5):
        model.train()
        run_epoch(data_gen(V, 30, 20), model,
                  SimpleLossCompute(model.generator, criterion, model_opt),device)
        # model.eval()
        # print(run_epoch(data_gen(V, 30, 5), model,
        #                 SimpleLossCompute(model.generator, criterion, None),device))
    model.eval()
    src = Variable(torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))
    src_mask = Variable(torch.ones(1, 1, 10))
    print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))