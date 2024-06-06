import torch
from torch import nn
from torchcrf import CRF
import torch.optim as optim

# class Config(object):
#     def __init__(self, vocab_size, embed_dim, label_num):
#         self.model_name = 'TextLSTM'
#         self.vocab_size = vocab_size
#         self.embed_dim = embed_dim
#         self.label_num = label_num
#         self.hidden_size = 256
#         self.num_layer = 3
#         self.dropout = 0.2
#         self.lr = 0.0001
#         self.num_tags=23
#
#
# class Model(torch.nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.embedding = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=config.vocab_size - 1)
#         self.lstm = nn.LSTM(config.embed_dim, config.hidden_size, config.num_layer,
#                             bidirectional=True, batch_first=True, dropout=config.dropout)
#         self.fc = nn.Linear(config.hidden_size * 2, config.label_num)
#
#     def forward(self, input):
#         # input: batchsize,seq_length = 128, 50
#         embed = self.embedding(input)
#         # embed: batchsize,seq_length,embed_dim = 128, 50, 300
#         hidden, _ = self.lstm(embed)
#         # hidden=self.crf(hidden)
#         # hidden: batchsize, seq, embedding = 128, 50, 256
#         hidden = hidden[:, -1, :]
#         # hidden: batchsize, seq_embedding = 128, 256
#         logit = torch.sigmoid(self.fc(hidden))
#         # logit: batchsize, label_logit = 128, 10
#         return logit

START_TAG = "<START>"
STOP_TAG = "<STOP>"

class LSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(LSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        forward_var = init_alphas

        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(self._log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = self._log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags):
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []

            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = self._argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + \
            self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = self._argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]
        best_path.reverse()
        return path_score, best_path

    def _log_sum_exp(self, vec):
        max_score = vec[0, self._argmax(vec)]
        max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
        return max_score + \
            torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

    def _argmax(self, vec):
        _, idx = torch.max(vec, 1)
        return idx.item()

    def neg_log_likelihood(self, sentence, tags):
        self.hidden = self.init_hidden()
        embeds = self.word_embeddings(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        tag_scores = self.hidden2tag(lstm_out)

        forward_score = self._forward_alg(tag_scores)
        gold_score = self._score_sentence(tag_scores, tags)
        return forward_score - gold_score

    def forward(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeddings(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        tag_scores = self.hidden2tag(lstm_out)

        score, tag_seq = self._viterbi_decode(tag_scores)
        return score, tag_seq
