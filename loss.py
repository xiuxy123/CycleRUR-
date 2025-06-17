# modified from https://github.com/maszhongming/MatchSum
from typing import List

import torch
from torch import nn
from transformers import T5Model, RobertaModel, T5Tokenizer
import torch
import torch.nn.functional as F


def RankingLoss(score, summary_score=None, margin=0.2, gold_margin=0.2, gold_weight=1, no_gold=False, no_cand=False):
    ones = torch.ones_like(score)
    loss_func = torch.nn.MarginRankingLoss(0.0)
    TotalLoss = loss_func(score, score, ones)
    # candidate loss
    n = score.size(1)
    if not no_cand:
        for i in range(1, n):
            pos_score = score[:, :-i]
            neg_score = score[:, i:]
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)
            ones = torch.ones_like(pos_score)
            loss_func = torch.nn.MarginRankingLoss(margin * i)
            loss = loss_func(pos_score, neg_score, ones)
            TotalLoss += loss
    if no_gold:
        return TotalLoss
    # gold summary loss
    pos_score = summary_score.unsqueeze(1).expand_as(score)
    neg_score = score
    pos_score = pos_score.contiguous().view(-1)
    neg_score = neg_score.contiguous().view(-1)
    ones = torch.ones_like(pos_score)
    loss_func = torch.nn.MarginRankingLoss(gold_margin)
    TotalLoss += gold_weight * loss_func(pos_score, neg_score, ones)
    return TotalLoss


class ReRanker(nn.Module):
    def __init__(self, encoder=None, tokenizer=None, args=None, pad_token_id=None):
        super(ReRanker, self).__init__()
        self.pad_token_id = pad_token_id
        self.args = args
        # self.model = T5Model.from_pretrained(encoder)
        # self.tokenizer = T5Tokenizer.from_pretrained(encoder)
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.vocab_size = 32128

    def text_encoder(self, text_input, input_mask):

        input_ids = text_input
        attention_mask = text_input != self.pad_token_id

        # attention_mask = torch.ones_like(input_ids)
        # input_tensor = torch.randint(0, self.vocab_size, (text_input.shape[0], text_input.shape[0])).to(text_input.device)

        # decoder_input_ids = torch.tensor([[self.model.config.decoder_start_token_id]]).to(text_input.device)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']

        return outputs

    def get_score(self, emb_1, emb_2, max_length):
        # 获取两个嵌入张量的维度数
        num_dims_1 = emb_1.dim()
        num_dims_2 = emb_2.dim()

        # 确保两个张量的维度数相同
        assert num_dims_1 == num_dims_2, "emb_1 and emb_2 must have the same number of dimensions"


        if num_dims_1 == 3:

            emb_1_pad = torch.zeros((emb_2.shape[0], max_length, emb_2.shape[2])).to(emb_2.device)
            # 将原始张量复制到填充张量的适当位置
            emb_1_pad[:emb_1.size(0), :emb_1.size(1), :emb_1.size(2)] = emb_1

            # 创建一个填充张量
            emb_2_pad = torch.zeros((emb_2.shape[0], max_length, emb_2.shape[2])).to(emb_2.device)
            # 将原始张量复制到填充张量的适当位置
            emb_2_pad[:emb_2.size(0), :emb_2.size(1), :emb_2.size(2)] = emb_2

            # 计算余弦相似度，注意这里我们需要对最后一个维度（embedding_dim）进行操作
            summary_score = torch.cosine_similarity(emb_1_pad, emb_2_pad, dim=-1)

        else:
            emb_1_pad = torch.zeros((emb_1.shape[0], emb_1.shape[1], max_length, emb_1.shape[3])).to(emb_1.device)
            # 将原始张量复制到填充张量的适当位置
            emb_1_pad[:emb_1.size(0), :emb_1.size(1), :emb_1.size(2), :emb_1.size(3)] = emb_1

            # 创建一个填充张量
            emb_2_pad = torch.zeros((emb_2.shape[0], emb_2.shape[1], max_length, emb_2.shape[3])).to(emb_2.device)
            # 将原始张量复制到填充张量的适当位置
            emb_2_pad[:emb_2.size(0), :emb_2.size(1), :emb_2.size(2), :emb_2.size(3)] = emb_2

            # 计算余弦相似度，注意这里我们需要对最后一个维度（embedding_dim）进行操作
            summary_score = torch.cosine_similarity(emb_1_pad, emb_2_pad, dim=-1)


        return summary_score

    def forward(self, text_id, candidate_id, summary_id=None, require_gold=True):

        batch_size = candidate_id.size(0)

        # input_mask = text_id != self.pad_token_id
        # out = self.text_encoder(text_id, input_mask)

        # processed_labels = self.tokenizer(text_id, return_tensors="pt", padding=True, truncation=True,
        #                                   max_length=self.args.max_input_length)
        #
        # encoder_input = {k: v.to(candidate_id.device) for k, v in processed_labels.items()}
        processed_labels = self.tokenizer(summary_id, return_tensors="pt", padding=True, truncation=True,
                                          max_length=self.args.max_input_length)

        input_mask = text_id != self.pad_token_id
        processed_labels['input_ids'] = text_id
        processed_labels['attention_mask'] = input_mask
        encoder_input = {k: v.to(candidate_id.device) for k, v in processed_labels.items()}

        out = self.encoder(**encoder_input)['last_hidden_state']

        doc_emb = out



        if require_gold:
            # get reference score
            # input_mask = summary_id != self.pad_token_id
            # processed_labels['input_ids'] = summary_id
            # processed_labels['attention_mask'] = input_mask
            # encoder_input = {k: v.to(candidate_id.device) for k, v in processed_labels.items()}

            processed_labels = self.tokenizer(summary_id, return_tensors="pt", padding=True, truncation=True,
                                              max_length=self.args.max_input_length)

            encoder_input = {k: v.to(candidate_id.device) for k, v in processed_labels.items()}

            out = self.encoder(**encoder_input)['last_hidden_state']
            summary_emb = out

            max_length = max(doc_emb.shape[-2], candidate_id.shape[-1], summary_emb.shape[-2])

            summary_score = self.get_score(summary_emb, doc_emb, max_length=max_length)

        candidate_num = candidate_id.size(1)
        candidate_id = candidate_id.view(-1, candidate_id.size(-1))
        input_mask = candidate_id != self.pad_token_id

        processed_labels['input_ids'] = candidate_id
        processed_labels['attention_mask'] = input_mask
        encoder_input = {k: v.to(candidate_id.device) for k, v in processed_labels.items()}
        out = self.encoder(**encoder_input)['last_hidden_state']

        # out = self.text_encoder(candidate_id, attention_mask=input_mask)
        candidate_emb = out.view(batch_size, out.shape[0] // batch_size, candidate_id.shape[1], -1)

        # get candidate score
        doc_emb = doc_emb.unsqueeze(1)
        score = self.get_score(candidate_emb, doc_emb, max_length=max_length)

        output = {'score': score}
        if require_gold:
            output['summary_score'] = summary_score


        loss = RankingLoss(output['score'], output['summary_score'])

        return loss


class Wordsloss(nn.Module):
    def __init__(self, encoder, tokenizer, args, ignore_index=-100, margin=0.2, temperature=1.0, device=None):
        super(Wordsloss, self).__init__()
        self.margin = margin
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.args = args
        self.temperature = temperature
        self.ignore_index = ignore_index
        self.device = device

    def compute_gl_loss(self, graph_1, graph_2):
        # 计算相似性矩阵

        batch_size_local, num_nodes_local, hidden_dim_local = graph_1.size()
        batch_size_global, num_nodes_global, hidden_dim_global = graph_2.size()

        max_length = max(graph_1.shape[1], graph_2.shape[1])

        graph_1_pad = torch.zeros((graph_1.shape[0], max_length, graph_1.shape[2])).to(graph_1.device)
        # 将原始张量复制到填充张量的适当位置
        graph_1_pad[:graph_1.size(0), :graph_1.size(1), :graph_1.size(2)] = graph_1

        # 创建一个填充张量
        graph_2_pad = torch.zeros((graph_1.shape[0], max_length, graph_2.shape[2])).to(graph_2.device)
        # 将原始张量复制到填充张量的适当位置
        graph_2_pad[:graph_2.size(0), :graph_2.size(1), :graph_2.size(2)] = graph_2


        # 计算相似性矩阵
        similarities = torch.matmul(graph_1_pad, graph_2_pad.transpose(1, 2))

        # 构建正样本对和负样本对的索引
        pos_pairs = [(i, i) for i in range(num_nodes_global)]  # 正样本对，同一个节点的局部和全局表示
        neg_pairs = [(i, i + 1) if i != num_nodes_global - 1 else (i, i - 1) for i in range(num_nodes_global)]  # 负样本对，不同节点的局部和全局表示

        pos_pairs_tensor = torch.LongTensor(pos_pairs).to(similarities.device)
        neg_pairs_tensor = torch.LongTensor(neg_pairs).to(similarities.device)

        # 计算正样本对的相似性和负样本对的相似性
        pos_similarities = similarities[:, pos_pairs_tensor[:, 0], pos_pairs_tensor[:, 1]]
        neg_similarities = similarities[:, neg_pairs_tensor[:, 0], neg_pairs_tensor[:, 1]]

        # 计算对比学习损失
        targets = torch.ones_like(pos_similarities)
        loss = F.margin_ranking_loss(pos_similarities, neg_similarities, targets, margin=self.margin, reduction='mean')

        return loss

    def compute_src_loss(self, graph_1, graph_2, graph_src):


        max_length = max(graph_1.shape[1], graph_2.shape[1], graph_src.shape[1])

        graph_1_pad = torch.zeros((graph_2.shape[0], max_length, graph_2.shape[2])).to(graph_1.device)
        # 将原始张量复制到填充张量的适当位置
        graph_1_pad[:graph_1.size(0), :graph_1.size(1), :graph_1.size(2)] = graph_1

        # 创建一个填充张量
        graph_2_pad = torch.zeros((graph_2.shape[0], max_length, graph_2.shape[2])).to(graph_2.device)
        # 将原始张量复制到填充张量的适当位置
        graph_2_pad[:graph_2.size(0), :graph_2.size(1), :graph_2.size(2)] = graph_2

        # 创建一个填充张量
        graph_src_pad = torch.zeros((graph_2.shape[0], max_length, graph_2.shape[2])).to(graph_2.device)
        # 将原始张量复制到填充张量的适当位置
        graph_src_pad[:graph_src.size(0), :graph_src.size(1), :graph_src.size(2)] = graph_src

        positive_embedding = torch.mean(graph_1_pad,dim=1)
        negative_embedding = torch.mean(graph_2_pad,dim=1)
        real_embedding = torch.mean(graph_src_pad,dim=1)

        # pos_sim = torch.cosine_similarity(graph_1_pad, graph_src_pad)
        # neg_sim = torch.cosine_similarity(graph_2_pad, graph_src_pad)

        positive_cos = torch.cosine_similarity(positive_embedding, real_embedding)
        negative_cos = torch.cosine_similarity(negative_embedding, real_embedding)

        positive_cos = positive_cos.unsqueeze(dim=-1)
        negative_cos = negative_cos.unsqueeze(dim=-1)


        pos_exp = positive_cos.exp()
        neg_exp = negative_cos.exp().sum(dim=-1)
        # # neg_exp = negative_cos.exp()
        loss = (- torch.log(pos_exp / (pos_exp + neg_exp))).mean()

        # targets = torch.ones_like(positive_cos)
        #
        # loss = torch.margin_ranking_loss(positive_cos, negative_cos, targets, margin=self.margin, reduction=1)

        return loss

    def forward(self, candidate_1, candidate_2, src_graph=None):

        processed_labels = self.tokenizer(candidate_2, return_tensors="pt", padding=True, truncation=True,
                                          max_length=self.args.max_input_length)

        encoder_input_2 = {k: v.to(self.device) for k, v in processed_labels.items()}

        graph_2 = self.encoder(**encoder_input_2)['last_hidden_state']

        # src-graph
        loss = None
        if src_graph is not None:
            processed_labels = self.tokenizer(src_graph, return_tensors="pt", padding=True, truncation=True,max_length=self.args.max_input_length)

            encoder_input_src = {k: v.to(self.device) for k, v in processed_labels.items()}


            graph_src = self.encoder(**encoder_input_src)['last_hidden_state']

            loss = self.compute_src_loss(graph_2, graph_src, candidate_1)

        else:
            processed_labels = self.tokenizer(candidate_1, return_tensors="pt", padding=True, truncation=True,
                                              max_length=self.args.max_input_length)

            encoder_input_1 = {k: v.to(self.device) for k, v in processed_labels.items()}

            graph_1 = self.encoder(**encoder_input_1)['last_hidden_state']

            loss = self.compute_gl_loss(graph_2, graph_1)

        return loss


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))  # 将 one_hot 张量注册为模型的缓冲区，以便在模型的前向传播中使用。

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target, average=True):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1).to(output.device)
        # 假设 target 是一个一维张量，并且它包含从 0 到 model_prob.size(1) - 1 的索引

        assert all(target >= 0) and all(target < model_prob.size(1)), "Index values out of bounds"
        # 确保 target 的数据类型是 long
        target = target.long()

        model_prob.scatter_(1, target.unsqueeze(1).to(output.device), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1).to(output.device), 0)

        log_probs = F.log_softmax(output, dim=-1)

        if average:
            return F.kl_div(log_probs, model_prob, reduction='batchmean')
        else:
            # (batch_size,)
            return F.kl_div(log_probs, model_prob, reduction='none').sum(1)


        # src_size = encoder_input_2['input_ids'].size()
        #
        # label_pos = torch.zeros(size=(src_size[0], src_size[1]), device=candidate_1.device)
        # label_neg = torch.ones(size=(src_size[0], src_size[1]), device=candidate_1.device)
        #
        # candidate_2_pos = []
        # # candidate_2_neg = []
        # for i in range(candidate_1.shape[0]):
        #     word_list_pos = []
        #     long_array = encoder_input_2['input_ids'][i]
        #     for word in candidate_2[i].split(" "):
        #
        #         if word in src_graph[i].split(" "):
        #             processed_labels = self.tokenizer(word, return_tensors="pt", padding=True,
        #                                                   truncation=True,
        #                                                   max_length=self.args.max_input_length)
        #
        #             encoder_input_2_pos = {k: v.to(candidate_1.device) for k, v in processed_labels.items()}
        #             short_array_pos = encoder_input_2_pos['input_ids'][0, 0:-1]
        #             for j, item in enumerate(long_array):
        #                 if all(x == y for x, y in zip(long_array[j:j + len(short_array_pos)], short_array_pos)):
        #                     label_pos[i, j:j + len(short_array_pos)] = 1
        #                     label_neg[i, j:j + len(short_array_pos)] = 0
        #     # candidate_2_pos.append(word_list_pos)
        #
        # # print(label_pos.bool(),  label_neg.bool())
        #
        # positive = torch.masked_fill(encoder_input_2['input_ids'], label_neg.bool(), 1e-6)
        # negative = torch.masked_fill(encoder_input_2['input_ids'], label_pos.bool(), 1e-6)
        #
        # encoder_input_2['input_ids'] = positive
        # graph_1 = self.encoder(**encoder_input_2)['last_hidden_state']
        #
        # encoder_input_2['input_ids'] = negative
        # graph_2 = self.encoder(**encoder_input_2)['last_hidden_state']