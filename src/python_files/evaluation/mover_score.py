# Reference: https://github.com/Yale-LILY/SummEval/blob/master/evaluation/summ_eval/mover_score_metric.py
from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import string
import os
from pyemd import emd, emd_with_flow
from torch import nn
from math import log
from itertools import chain

from collections import defaultdict, Counter
from multiprocessing import Pool
from functools import partial

from transformers import AutoTokenizer, AutoModel

class MoverScore:
    """
    Customized MoverScore class from https://github.com/AIPHES/emnlp19-moverscore/blob/master/moverscore_v2.py
    """
    def __init__(self, model_name='distilbert-base-multilingual-cased') -> None:
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
        self.model.eval()

    def truncate(self, tokens):
        if len(tokens) > self.tokenizer.model_max_length - 2:
            tokens = tokens[0:(self.tokenizer.model_max_length - 2)]
        return tokens

    def process(self, a):
        a = ["[CLS]"] + self.truncate(tokenizer.tokenize(a)) + ["[SEP]"]
        a = self.tokenizer.convert_tokens_to_ids(a)
        return set(a)

    @staticmethod
    def get_idf_dict(arr, nthreads=4):
        idf_count = Counter()
        num_docs = len(arr)

        process_partial = partial(process)

        with Pool(nthreads) as p:
            idf_count.update(chain.from_iterable(p.map(process_partial, arr)))

        idf_dict = defaultdict(lambda: log((num_docs + 1) / (1)))
        idf_dict.update({idx: log((num_docs + 1) / (c + 1)) for (idx, c) in idf_count.items()})
        return idf_dict

    def padding(self, arr, pad_token, dtype=torch.long):
        lens = torch.LongTensor([len(a) for a in arr])
        max_len = lens.max().item()
        padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
        mask = torch.zeros(len(arr), max_len, dtype=torch.long)
        for i, a in enumerate(arr):
            padded[i, :lens[i]] = torch.tensor(a, dtype=dtype)
            mask[i, :lens[i]] = 1
        return padded, lens, mask

    @staticmethod
    def bert_encode(model, x, attention_mask):
        model.eval()
        with torch.no_grad():
            result = model(x, attention_mask=attention_mask)
        return result.hidden_states

    def collate_idf(self, arr, tokenize, numericalize, idf_dict, pad="[PAD]"):
        tokens = [["[CLS]"] + self.truncate(tokenize(a)) + ["[SEP]"] for a in arr]
        arr = [numericalize(a) for a in tokens]

        idf_weights = [[idf_dict[i] for i in a] for a in arr]

        pad_token = numericalize([pad])[0]

        padded, lens, mask = self.padding(arr, pad_token, dtype=torch.long)
        padded_idf, _, _ = self.padding(idf_weights, pad_token, dtype=torch.float)

        return padded, padded_idf, lens, mask, tokens

    def get_bert_embedding(self, all_sens, model, tokenizer, idf_dict,
                           batch_size=-1):
        padded_sens, padded_idf, lens, mask, tokens = self.collate_idf(all_sens,
                                                                  tokenizer.tokenize, tokenizer.convert_tokens_to_ids,
                                                                  idf_dict)
        if batch_size == -1: batch_size = len(all_sens)

        embeddings = []
        with torch.no_grad():
            for i in range(0, len(all_sens), batch_size):
                batch_embedding = self.bert_encode(model, padded_sens[i:i + batch_size],
                                              attention_mask=mask[i:i + batch_size])

                batch_embedding = torch.stack(batch_embedding)

                embeddings.append(batch_embedding)
                del batch_embedding

        total_embedding = torch.cat(embeddings, dim=-3)

        return total_embedding, lens, mask, padded_idf, tokens

    @staticmethod
    def _safe_divide(numerator, denominator):
        return numerator / (denominator + 1e-30)

    @staticmethod
    def batched_cdist_l2(x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.baddbmm(
            x2_norm.transpose(-2, -1),
            x1,
            x2.transpose(-2, -1),
            alpha=-2
        ).add_(x1_norm).clamp_min_(1e-30).sqrt_()
        return res

    def word_mover_score(self, refs, hyps, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=True,
                         batch_size=256):
        preds = []
        for batch_start in range(0, len(refs), batch_size):
            batch_refs = refs[batch_start:batch_start + batch_size]
            batch_hyps = hyps[batch_start:batch_start + batch_size]

            ref_embedding, ref_lens, ref_masks, ref_idf, ref_tokens = self.get_bert_embedding(batch_refs, self.model, self.tokenizer,
                                                                                         idf_dict_ref)
            hyp_embedding, hyp_lens, hyp_masks, hyp_idf, hyp_tokens = self.get_bert_embedding(batch_hyps, self.model, self.tokenizer,
                                                                                         idf_dict_hyp)

            # get last model layer as embedding
            ref_embedding = ref_embedding[-1]
            hyp_embedding = hyp_embedding[-1]

            # power mean
            #         ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
            #         hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))

            #         ref_embedding_max, _ = torch.max(ref_embedding[-5:], dim=0, out=None)
            #         hyp_embedding_max, _ = torch.max(hyp_embedding[-5:], dim=0, out=None)

            #         ref_embedding_min, _ = torch.min(ref_embedding[-5:], dim=0, out=None)
            #         hyp_embedding_min,_ = torch.min(hyp_embedding[-5:], dim=0, out=None)

            #         ref_embedding_avg = ref_embedding[-5:].mean(0)
            #         hyp_embedding_avg = hyp_embedding[-5:].mean(0)

            #         ref_embedding = torch.cat([ref_embedding_min, ref_embedding_avg, ref_embedding_max], -1)
            #         hyp_embedding = torch.cat([hyp_embedding_min, hyp_embedding_avg, hyp_embedding_max], -1)

            # Remove stop words, token ##, and punctuation
            batch_size = len(ref_tokens)
            for i in range(batch_size):
                ref_ids = [k for k, w in enumerate(ref_tokens[i])
                           if w in stop_words or '##' in w
                           or w in set(string.punctuation)]
                hyp_ids = [k for k, w in enumerate(hyp_tokens[i])
                           if w in stop_words or '##' in w
                           or w in set(string.punctuation)]

                ref_embedding[i, ref_ids, :] = 0
                hyp_embedding[i, hyp_ids, :] = 0
                ref_idf[i, ref_ids] = 0
                hyp_idf[i, hyp_ids] = 0

            raw = torch.cat([ref_embedding, hyp_embedding], 1)

            raw.div_(torch.norm(raw, dim=-1).unsqueeze(-1) + 1e-30)

            distance_matrix = self.batched_cdist_l2(raw, raw).double().cpu().numpy()

            for i in range(batch_size):
                c1 = np.zeros(raw.shape[1], dtype=np.float)
                c2 = np.zeros(raw.shape[1], dtype=np.float)
                c1[:len(ref_idf[i])] = ref_idf[i]
                c2[len(ref_idf[i]):] = hyp_idf[i]

                c1 = self._safe_divide(c1, np.sum(c1))
                c2 = self._safe_divide(c2, np.sum(c2))

                dst = distance_matrix[i]
                _, flow = emd_with_flow(c1, c2, dst)
                flow = np.array(flow, dtype=np.float32)
                score = 1. / (1. + np.sum(flow * dst))  # 1 - np.sum(flow * dst)
                preds.append(score)

        return preds


class MyMoverScore:
    """
    My Mover Score Functions
    if called append to batch and if batch size reached compute batch
    """
    def __init__(self, mover_score_class, batch_size=16) -> None:
        """
        Init
        :param mover_score_class: to compute Moverscore
        :type mover_score_class: MoverScore class
        :param batch_size: batch size
        :type batch_size: int
        """
        super().__init__()
        # self.mover_score_metric = MoverScoreMetric()
        self.count_batch = 0
        self.batch_size = batch_size
        self.batch_target = []
        self.batch_prediction = []
        self.mover_score = []
        self.mover_score_class = mover_score_class

    def __call__(self, target, prediction):
        """
        Call for MoverScore computations
        :param target: target text
        :type target: str
        :param prediction: predicted text
        :type prediction: str
        """
        self.batch_target.append(target)
        self.batch_prediction.append(prediction)

        if self.count_batch == self.batch_size:
            self.mover_score.append(self.evaluate_batch(self.batch_target, self.batch_prediction))
            self.batch_target = []
            self.batch_prediction = []
            self.count_batch = 0
        else:
            self.count_batch += 1

    def evaluate_batch(self, summaries, references):
        idf_dict_summ = defaultdict(lambda: 1.)
        idf_dict_ref = defaultdict(lambda: 1.)
        scores = self.mover_score_class.word_mover_score(references, summaries, idf_dict_ref, idf_dict_summ, \
                                  stop_words=[], remove_subwords=True, \
                                  batch_size=self.batch_size)
        return np.mean(scores)

    def result(self):
        """
        Get results
        :return: result MoverScore
        :rtype: float
        """
        if self.count_batch != 0:
            self.mover_score.append(self.evaluate_batch(self.batch_target, self.batch_prediction))
            self.batch_target = []
            self.batch_prediction = []
            self.count_batch = 0

        score_list = []
        for score in self.mover_score:
            score_list.append(score)
        mover_score = np.mean(np.array(score_list))
        print("Moverscore: {:.2f}".format(mover_score))
        return mover_score