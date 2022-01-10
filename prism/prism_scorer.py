#!/usr/bin/env python3

import argparse

import os
import sys
import warnings
from typing import List, Dict, Iterator, Any, Tuple

import numpy as np
import sentencepiece as spm
import torch
from fairseq import checkpoint_utils, utils
from fairseq.data import LanguagePairDataset
from sacrebleu import get_source_file, get_reference_files, DATASETS, get_langpairs_for_testset

from prism.sequence_scorer import SequenceScorer
from prism.models import MODELS, hash_model, MODEL_DIR, PrismDataError

import logging
logger = logging.getLogger("prism")
logger.setLevel(logging.INFO)

class Prism:
    def __init__(self, lang, temperature=1.0):
        '''
        model_dir should contain:
         1) checkpoint.pt: the fairseq model
         2) spm.model: the sentencepiece model
         3) dict.src.txt: the fairseq source dictionary
         4) dict.tgt.txt: the fairseq target dictionary (likely a copy of the source)

        lang: ISO 639-1 Code (e.g. "en"). Must be a language compatable with the model.
        '''
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(MODEL_DIR + '/spm.model')

        self.lang = lang
        self.temperature = temperature

        # this prints things and I can't figure out how to disable it
        sys.stdout = open(os.devnull, 'w')
        self.models, self.args, self.task = checkpoint_utils.load_model_ensemble_and_task(
            [MODEL_DIR + '/checkpoint.pt', ],
            arg_overrides=dict(data=MODEL_DIR + '/'),
        )
        sys.stdout = sys.__stdout__

        self.use_cuda = torch.cuda.is_available()

        self.generator = SequenceScorer(self.task.target_dictionary, temperature=temperature)

        for model in self.models:
            if self.use_cuda:
                model.cuda()
            model.make_generation_fast_(
                beamable_mm_beam_size=None,
                need_attn=False,
            )
            #if fp16:
            #    model.half()

        # hash model
        self.model_hash = hash_model(MODEL_DIR)

        if self.model_hash in MODELS:
            model_langs = MODELS[self.model_hash]['langs']
            if lang not in model_langs:
                model_name = MODELS[self.model_hash]['name']
                raise PrismDataError(f'Language "{lang}" is unsupported for model "{model_name}"')
                raise PrismDataError(f'Supported languages for {model_name}: {", ".join(model_langs)}')
        else:
            raise PrismDataError('Unrecognized model, so cannot check language')

    def identifier(self):
        if self.model_hash in MODELS:
            model_name = MODELS[self.model_hash]['name']
        else:
            warnings.warn('unrecognized model, using hash to identify')
            model_name = self.model_hash

        return dict(version='0.1', model=model_name, seg_scores='avg_log_prob',
                    sys_scores='avg_log_prob', log_base=2, temperature=self.temperature)

    def _binarize(self, sentence: str) -> torch.LongTensor:
        return self.task.source_dictionary.encode_line(sentence, add_if_not_exist=False).long()

    def _encode(self, sent, prepend=True):
        sent = ' '.join(self.sp.EncodeAsPieces(sent))
        if prepend:
            sent = f'<{self.lang}> ' + sent
        return self._binarize(sent)

    def _build_batches(self,
                       source_tokens: List[List[int]],
                       target_tokens: List[List[int]],
                       skip_invalid_size_inputs: bool) -> Iterator[Dict[str, Any]]:
        source_lengths = torch.LongTensor([t.numel() for t in source_tokens])
        target_lengths = torch.LongTensor([t.numel() for t in target_tokens])

        batch_iterator = self.task.get_batch_iterator(
            dataset=LanguagePairDataset(source_tokens, source_lengths, self.task.source_dictionary,
                                        tgt=target_tokens, tgt_sizes=target_lengths,
                                        tgt_dict=self.task.target_dictionary),
            max_tokens=self.args.max_tokens,
            max_sentences=self.args.max_sentences,
            max_positions=(2000, 2000),  # ???
            ignore_invalid_inputs=skip_invalid_size_inputs,
        ).next_epoch_itr(shuffle=False)
        return batch_iterator

    def _score_forward(self, tok_sents_in, tok_sents_out):
        assert len(tok_sents_in) == len(tok_sents_out)
        tok_level_scores = [None, ] * len(tok_sents_in)  # for debug
        results = [None, ] * len(tok_sents_in)
        for batch in self._build_batches(tok_sents_in, tok_sents_out, skip_invalid_size_inputs=False):
            if self.use_cuda:  # must be a better way
                batch['id'] = batch['id'].cuda()
                batch['net_input']['src_tokens'] = batch['net_input']['src_tokens'].cuda()
                batch['net_input']['src_lengths'] = batch['net_input']['src_lengths'].cuda()
                batch['net_input']['prev_output_tokens'] = batch['net_input']['prev_output_tokens'].cuda()
                batch['target'] = batch['target'].cuda()

            translations = self.task.inference_step(self.generator, self.models, batch)

            ids = batch['id'].cpu().numpy()

            tok_scores = [x[0]['positional_scores'].cpu().numpy() for x in translations]

            # [1:] to skip language tag log prob
            sent_scores = [np.mean(x[1:]) for x in tok_scores]

            for _id, sent_score, _tok_score in zip(ids, sent_scores, tok_scores):
                results[_id] = sent_score
                tok_level_scores[_id] = _tok_score

        if logger.level == logging.DEBUG:
            for ii, (sent_in, scores_out, sent_out) in enumerate(zip(tok_sents_in, tok_level_scores, tok_sents_out)):
                sent_in_str = ' '.join([self.task.source_dictionary[x] for x in sent_in])
                logger.debug(f'Input[{ii}] = ' + sent_in_str)
                sent_out_tok = [self.task.source_dictionary[x] for x in sent_out]
                logger.debug(f'Output[{ii}] = ' + \
                             f' '.join([f'{a}[{b:.02f}]' for a, b in zip(sent_out_tok, scores_out)]))

        if None in results:
            raise Exception('Missing one or more sentence scores')

        return np.array(results)

    def score(self, cand, ref=None, src=None, segment_scores=False):

        if not (ref is None) ^ (src is None):
            raise Exception('Must provide exactly one of "ref" or "src"')

        tokenized_cand = [self._encode(sentence, prepend=False) for sentence in cand]
        tokenized_cand_prep = [self._encode(sentence, prepend=True) for sentence in cand]

        if src is not None:
            # Prism-src: score candidate given on source
            if len(cand) != len(src):
                raise Exception(f'Length of cand ({len(cand)}) does not match length of src ({len(src)})')
            tokenized_src = [self._encode(sentence, prepend=False) for sentence in src]
            scores = self._score_forward(tokenized_src, tokenized_cand_prep)

        else:
            # Prism-ref: average candidate given reference and reference given candidate
            if len(cand) != len(ref):
                raise Exception(f'Length of cand ({len(cand)}) does not match length of ref ({len(ref)})')
            tokenized_ref = [self._encode(sentence, prepend=False) for sentence in ref]
            tokenized_ref_prep = [self._encode(sentence, prepend=True) for sentence in ref]
            forward_scores = self._score_forward(tok_sents_in=tokenized_ref, tok_sents_out=tokenized_cand_prep)
            reverse_scores = self._score_forward(tok_sents_in=tokenized_cand, tok_sents_out=tokenized_ref_prep)
            scores = 0.5 * forward_scores + 0.5 * reverse_scores

        if not segment_scores:
            scores = np.mean(scores)

        return scores
