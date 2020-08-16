#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Translate pre-processed data with a trained model.
"""

import torch

from fairseq import bleu, checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter

from collections import defaultdict

import torch
from fairseq import utils
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


def make_subword_penalties(line):
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


def make_word_penalties(line, vocab, mapx):
    """
    prefix: subwords that make up a n-gram (n=1,2,3,4) of FULL WORDS
    penalize: the next subword
    """
    from time import time
    t0 = time()

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
    line2 = [x.replace('|', ' ').strip().split() for x in line2.replace(' ', '|').split(uline) if x]

    for prefix_len in (0,1,2,3):
        for ii in range(len(line2)-prefix_len):
            prefix = line2[ii:ii+prefix_len]
            next_word = uline + line2[ii+prefix_len][0]  # just penalize starting a word, not continuing it

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
                    
            for tok in mapx[longest_next_substring]:  # every word that starts the same, sans case
                penalties[tuple(word_prefix)].append( (tok, len(prefix)) )

    return penalties


def make_vocab_start_map(vocab):
    vocab_set = set(vocab)
    
    # build mapping from every lowercase subword to every cased variant in vocabulary
    ucase2case = defaultdict(set)
    for word in vocab:
        for ii in range(1,len(word)+1):
            subword = word[:ii]
            if subword in vocab_set:
                ucase2case[subword.lower()].add(subword)

    # build mapping from every word to every prefix that starts that word (where "starts" ignores case)
    mapx = dict()
    for word in vocab:
        toks = set()
        for ii in range(1,len(word)+1):
            subword = word[:ii]
            for fubar in ucase2case[subword.lower()]:
                toks.add(fubar)
        mapx[word] = list(toks)

    return mapx


class NgramDownweightEncoder(FairseqEncoder):

    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.mapx = make_vocab_start_map(self.dictionary.symbols)
        self.vocab_set = set(self.dictionary.symbols)
        self.args = args

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
            penalties = make_word_penalties(line=line, vocab=self.vocab_set, mapx=self.mapx)
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


############  /NEW
###################################################################

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

    try:
        print(args.print_step)
    except:
        args.print_step = False # why do I need this???

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Optimize ensemble for generation
    for model in models:
        
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    ngram_downweight_model = NgramDownweightModel.build_model(args, task)  
    models.append(ngram_downweight_model)   # ensemble Prism multilingual NMT model and model to downweight n-grams

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    max_positions = [model.max_positions() for model in models]
    fixed_max_positions = []
    for x in max_positions:
        try:
            fixed_max_positions.append( (x[0], x[1]) )
        except:
            fixed_max_positions.append( (12345677, x) )

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),  *fixed_max_positions ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()
    generator = task.build_generator(args)

    # Generate and compute BLEU score
    if args.sacrebleu:
        scorer = bleu.SacrebleuScorer()
    else:
        scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
    num_sentences = 0
    has_target = True
    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()
        for sample in t:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if 'net_input' not in sample:
                continue

            prefix_tokens = None
            if args.prefix_size > 0:
                prefix_tokens = sample['target'][:, :args.prefix_size]

            gen_timer.start()
            hypos = task.inference_step(generator, models, sample, prefix_tokens)
            num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
            gen_timer.stop(num_generated_tokens)

            for i, sample_id in enumerate(sample['id'].tolist()):
                has_target = sample['target'] is not None

                # Remove padding
                src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
                target_tokens = None
                if has_target:
                    target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()

                # Either retrieve the original sentences or regenerate them from tokens.
                if align_dict is not None:
                    src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                    target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
                else:
                    if src_dict is not None:
                        src_str = src_dict.string(src_tokens, args.remove_bpe)
                    else:
                        src_str = ""
                    if has_target:
                        target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)

                if not args.quiet:
                    if src_dict is not None:
                        print('S-{}\t{}'.format(sample_id, src_str))
                    if has_target:
                        print('T-{}\t{}'.format(sample_id, target_str))

                # Process top predictions
                for j, hypo in enumerate(hypos[i][:args.nbest]):
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo['tokens'].int().cpu(),
                        src_str=src_str,
                        alignment=hypo['alignment'],
                        align_dict=align_dict,
                        tgt_dict=tgt_dict,
                        remove_bpe=args.remove_bpe,
                    )

                    if not args.quiet:
                        print('H-{}\t{}\t{}'.format(sample_id, hypo['score'], hypo_str))
                        print('P-{}\t{}'.format(
                            sample_id,
                            ' '.join(map(
                                lambda x: '{:.4f}'.format(x),
                                hypo['positional_scores'].tolist(),
                            ))
                        ))

                        if args.print_alignment:
                            print('A-{}\t{}'.format(
                                sample_id,
                                ' '.join(['{}-{}'.format(src_idx, tgt_idx) for src_idx, tgt_idx in alignment])
                            ))

                        if args.print_step:
                            print('I-{}\t{}'.format(sample_id, hypo['steps']))

                        if getattr(args, 'retain_iter_history', False):
                            print("\n".join([
                                    'E-{}_{}\t{}'.format(
                                        sample_id, step,
                                        utils.post_process_prediction(
                                            h['tokens'].int().cpu(),
                                            src_str, None, None, tgt_dict, None)[1])
                                        for step, h in enumerate(hypo['history'])]))

                    # Score only the top hypothesis
                    if has_target and j == 0:
                        if align_dict is not None or args.remove_bpe is not None:
                            # Convert back to tokens for evaluation with unk replacement and/or without BPE
                            target_tokens = tgt_dict.encode_line(target_str, add_if_not_exist=True)
                        if hasattr(scorer, 'add_string'):
                            scorer.add_string(target_str, hypo_str)
                        else:
                            scorer.add(target_tokens, hypo_tokens)

            wps_meter.update(num_generated_tokens)
            t.log({'wps': round(wps_meter.avg)})
            num_sentences += sample['nsentences']

    print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
    if has_target:
        print('| Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string()))

    return scorer


def cli_main():
    parser = options.get_generation_parser()

    # add options for Prism paraphrase generation
    parser.add_argument('--prism_a', type=float, help='prism_a ** prism_b. reasonable starting point: 0.003', required=True)
    parser.add_argument('--prism_b', type=float, help='prism_a ** prism_b. reasonable starting point: 4.0', required=True) 
    
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
