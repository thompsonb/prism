
# Generating Paraphrases with the Prism Model

This README describes how to generate paraphrases from the Prism model
using the algorithm described in [this paper](https://arxiv.org/abs/2008.04935).
You should have already installed the Prism environment and downloaded the Prism model -- see the [main README](../README.md). 


Make an example input file and apply sentencepiece. 
In python:
```python
import os
import sentencepiece as spm

lang='en'

sp = spm.SentencePieceProcessor()
sp.Load(os.environ['MODEL_DIR'] + '/spm.model')
sents = ['Among other things, the developments in terms of turnover, employment, warehousing and prices are recorded.', ]
sp_sents = [' '.join(sp.EncodeAsPieces(sent)) for sent in sents]

with open('test.src', 'wt') as fout:
     for sent in sp_sents:
         fout.write(sent + '\n')

# we also need a dummy output file with the language tag
with open('test.tgt', 'wt') as fout:
     for sent in sp_sents:
         fout.write(f'<{lang}> \n')
```

Run fairseq preprocessing:
```bash
fairseq-preprocess --source-lang src --target-lang tgt  \
    --joined-dictionary  --srcdict $MODEL_DIR/dict.tgt.txt \
    --trainpref  test  --validpref test  --testpref test --destdir test_bin
```

Generate paraphrases (prism_a and prism_b correspond to alpha and beta in the paper):
```
python generate_paraphrases.py test_bin --batch-size 8 \
   --prefix-size 1 \
   --path $MODEL_DIR/checkpoint.pt \
   --prism_a 0.006 --prism_b 4  
```


Output is printed preceded with "H-*":
```
H-0     -0.21274660527706146    <en> ▁Among ▁other ▁things , ▁development s ▁regarding ▁turn over , ▁jobs , ▁war eho us ing ▁and ▁rates ▁are ▁recorded .
```


# Publications
If you use the Prism model for paraphrase generation, please cite the Prism paper as well as the paraphrase generation paper from the [main README](../README.md#publications).


