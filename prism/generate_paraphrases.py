#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Translate pre-processed data with a trained model.
"""

from collections import defaultdict

import torch

from fairseq import checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.models import FairseqEncoder, FairseqDecoder, FairseqEncoderDecoderModel


###################################################################
##  PRISM PARAPHRASER GENERATION CODE
##
##  Here we build a fairseq model that gets ensembled with the regular multilngual NMT model
##  The new model simply downweights any vocabulary item that begins a new word that
##     matches an n-gram in the input (case is ignored).
##
##  See https://arxiv.org/abs/2008.04935 for more details on the generation modification
##  See https://arxiv.org/abs/2004.14564 for information about the model we use


def make_subword_penalties(line):  #TODO: Function not used?
    """
    prefix: n-grams of subwords (n=1,2,3,4)
    penalize: the next subword
    """
    penalties = defaultdict(list)
    toks = line.replace('<pad>', '').split()
    for prefix_len in (0,1,2,3):
        for ii in range(len(toks)-prefix_len):
            prefix = toks[ii:ii+prefix_len]
            next_word = toks[ii+prefix_len]
            penalties[tuple(prefix)].append( (next_word, len(prefix)) )

    return penalties

class NgramDownweightEncoder(FairseqEncoder):

    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.mapx = self.make_vocab_start_map(self.dictionary.symbols)
        self.vocab_set = set(self.dictionary.symbols)
        self.args = args

    def make_word_penalties(self, line, vocab, mapx):
        """
        prefix: subwords that make up a n-gram (n=1,2,3,4) of FULL WORDS
        penalize: the next subword
        """
        penalties = defaultdict(list)
        uline = '▁'
        def breakup(tt):
            out = []
            for word in tt:
                for ii, subword in enumerate(word):
                    if ii == 0:
                        out.append(uline + subword)
                    else:
                        out.append(subword)
            return out
        line2 = line.replace('<pad>', '').strip()
        line2 = [x.replace('|', ' ').strip().split()
                 for x in line2.replace(' ', '|').split(uline) if x]

        for prefix_len in (0,1,2,3):
            for ii in range(len(line2)-prefix_len):
                prefix = line2[ii:ii+prefix_len]
                # just penalize starting a word, not continuing it
                next_word = uline + line2[ii+prefix_len][0]

                whole_next_word = uline + ''.join(line2[ii+prefix_len])

                word_prefix = breakup(prefix)

                # penalize any token that starts the next word, ignoring case
                # about 1s per line
                for tok in vocab:
                    if whole_next_word.lower().startswith(tok.lower()):
                        penalties[tuple(word_prefix)].append( (tok, len(prefix)) )

                # build the longest part of the next word I can that is in the vocab
                longest_next_substring = uline
                for subthing in line2[ii+prefix_len]:
                    if longest_next_substring+subthing in vocab:
                        longest_next_substring = longest_next_substring+subthing
                    else:
                        break

                for tok in mapx[longest_next_substring]:
                    # every word that starts the same, sans case
                    penalties[tuple(word_prefix)].append( (tok, len(prefix)) )

        return penalties

    def make_vocab_start_map(self, vocab):
        vocab_set = set(vocab)
        # build mapping from every lowercase subword to every cased
        # variant in vocabulary
        ucase2case = defaultdict(set)
        for word in vocab:
            for ii in range(1,len(word)+1):
                subword = word[:ii]
                if subword in vocab_set:
                    ucase2case[subword.lower()].add(subword)
        # build mapping from every word to every prefix that starts
        # that word (where "starts" ignores case)
        mapx = dict()
        for word in vocab:
            toks = set()
            for ii in range(1,len(word)+1):
                subword = word[:ii]
                for fubar in ucase2case[subword.lower()]:
                    toks.add(fubar)
            mapx[word] = list(toks)

        return mapx

    def forward(self, src_tokens, src_lengths):
        # Task, and in particular the ``'net_input'`` key in each
        # mini-batch. We discuss Tasks in the next tutorial, but for now just
        # know that *src_tokens* has shape `(batch, src_len)` and *src_lengths*
        # has shape `(batch)`.

        # Note that the source is typically padded on the left. This can be
        # configured by adding the `--left-pad-source "False"` command-line
        # argument, but here we'll make the Encoder handle either kind of
        # padding by converting everything to be right-padded.
        if self.args.left_pad_source:
            # Convert left-padding to right-padding.
            src_tokens = utils.convert_padding_direction(
                src_tokens,
                padding_idx=self.dictionary.pad(),
                left_to_right=True
            )

        # Return the Encoder's output. This can be any object and will be
        # passed directly to the Decoder.
        batch_size = src_tokens.shape[0]
        debug_out = self.dictionary.string(src_tokens,
                                           bpe_symbol=None, escape_unk=False)

        batch_penalties = []
        for line in debug_out.split('\n'):
            penalties = self.make_word_penalties(line=line, vocab=self.vocab_set, mapx=self.mapx)
            batch_penalties.append(penalties)

        return batch_penalties

    # Encoders are required to implement this method so that we can rearrange
    # the order of the batch elements during inference (e.g., beam search).
    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to `new_order`.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            `encoder_out` rearranged according to `new_order`
        """
        # these maps arent going to be modified, so multiple references is fine
        return [encoder_out[ii] for ii in new_order.cpu().numpy()]


class NgramDownweightDecoder(FairseqDecoder):

    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.args = args

    # During training Decoders are expected to take the entire target sequence
    # (shifted right by one position) and produce logits over the vocabulary.
    # The *prev_output_tokens* tensor begins with the end-of-sentence symbol,
    # ``dictionary.eos()``, followed by the target sequence.
    def forward(self, prev_output_tokens, encoder_out):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention

        Returns:
            tuple:
                - the last decoder layer's output of shape
                  `(batch, tgt_len, vocab)`
                - the last decoder layer's attention weights of shape
                  `(batch, tgt_len, src_len)`
        """

        batch_size = prev_output_tokens.shape[0]
        tgt_len = prev_output_tokens.shape[1]
        vocab_size = len(self.dictionary)

        xx = torch.zeros([batch_size, tgt_len, vocab_size], dtype=torch.float32)

        debug_out = self.dictionary.string(prev_output_tokens,
                                           bpe_symbol=None, escape_unk=False)

        lines = debug_out.split('\n')

        for ii, (line, penalties) in enumerate(zip(lines, encoder_out)):

            max_prefix_len = max([len(prefix) for prefix in penalties])

            #if ii == 0:
            #    print('#'*50)
            #    print('DECODE SO FAR:', line)

            toks = line.strip().split()
            #if ii == 0: print('input length:', len(toks))
            for n_gram_n in range(1, min(len(toks)+2, max_prefix_len)):
                prefix_size = n_gram_n - 1

                if prefix_size == 0:
                    prefix = tuple()
                else:
                    prefix = tuple(toks[-prefix_size:])

                if prefix in penalties:
                    for next_word, prefix_len_in_words in penalties[prefix]:
                        word_idx = self.dictionary.index(next_word)
                        xx[ii, -1, word_idx] -= self.args.prism_a * (prefix_len_in_words+1)  ** self.args.prism_b

        # Return the logits and ``None`` for the attention weights
        xx = xx.cuda()
        return xx, None


class NgramDownweightModel(FairseqEncoderDecoderModel):

    def max_positions(self):
        return (123456, 123456)

    @classmethod
    def build_model(cls, args, task):
        # Initialize our Encoder and Decoder.
        encoder = NgramDownweightEncoder(args=args, dictionary=task.source_dictionary)
        decoder = NgramDownweightDecoder(args=args, dictionary=task.target_dictionary)
        model = NgramDownweightModel(encoder, decoder)
        # Print the model architecture.
        #print(model)
        return model
    def get_normalized_probs(self, decoder_out, log_probs):
        return decoder_out[0]

class PrismParaphase:
    def __init__(self, a=0.003, b=4.0, *args, **kwargs):
        pass
    def generate(self, )


def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    print(args)




def cli_main():
    # Fairseq's options.
    parser = options.get_generation_parser()
    # add options for Prism paraphrase generation
    parser.add_argument('--prism_a', type=float, help='prism_a ** prism_b. reasonable starting point: 0.003', required=True)
    parser.add_argument('--prism_b', type=float, help='prism_a ** prism_b. reasonable starting point: 4.0', required=True)
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
