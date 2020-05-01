# Prism: MT Evaluation in Many Languages via Zero-Shot Paraphrasing

Prism is an automatic MT metric which uses a sequence-to-sequence
paraphraser to score MT system outputs conditioned on their respective
human references.  Prism uses a multilingual NMT model as a zero-shot
paraphraser, which negates the need for synthetic paraphrase data and
results in a single model which works in many languages.

Prism outperforms or statistically ties with all metrics submitted to
the [WMT 2019 metrics shared task](https://www.aclweb.org/anthology/W19-5302/) as
segment-level human correlation.

We provide a large, pre-trained multilingual NMT model which we use as a multilingual paraphraser, 
but the model may also be of use to the research community beyond MT metrics.

Prism scores detokenized, human-formatted inputs, and not tokenized ones.
All preprocessing is applied internally, making it particularly easy to use.
This document describes how to install and use prism.

# Installation

Prism requires a version of [Fairseq](https://github.com/pytorch/fairseq) compatible with the provided pretrained model.
You may wish to start with a clean environment:

```bash
conda create -n prismenv python=3.7 -y
conda activate prismenv  # older conda versions: source activate prismenv
```

For reasonable speeds, we recommend running on a machine with a GPU
and the [CUDA](https://developer.nvidia.com/cuda-zone) version compatible with the version of fairseq/torch installed above.
Prism will run on a GPU if available; to run on CPU instead, set CUDA_VISIBLE_DEVICES to an empty string.

Download the Prism code and install requirements, including Fairseq:

```bash
git clone https://github.com/thompsonb/prism
cd prism
pip install -r requirements.txt
```

# Download Model

```bash
wget http://data.statmt.org/prism/m39v1.tar
tar xf m39v1.tar
```

# Usage: Command Line

Create test candidate/reference files:

```bash
echo -e "Hi world.\nThis is a Test." >> cand.en
echo -e "Hello world.\nThis is a test." >> ref.en
echo -e "Bonjour le monde.\nC'est un test." >> src.fr
```

To obtain system-level metric scores, run:
```bash
./prism.py --cand cand.en --ref ref.en --lang en --model-dir m39v1/
```
Here, "ref.en" is the (untokenized) human reference, and "cand.en" is the (untokenized) system output.
This command will print some logging information to STDERR, including a model/version identifier,
and print the system-level score (negative, higher is better) to STDOUT:

>Prism identifier: {'version': '0.1', 'model': 'm39v1', 'seg_scores': 'avg_log_prob', 'sys_scores': 'avg_log_prob', 'log_base': 2}  
>-1.0184667  

Candidates can also be piped into prism.py:

```bash
cat cand.en | ./prism.py --ref ref.en --lang en --model-dir m39v1/
```

To score output using the source instead of the reference
(i.e., quality estimation as a metric), use the --src flag.
Note that --lang still specifies the target/reference language:

```bash
./prism.py --cand cand.en --src src.fr --lang en --model-dir m39v1/ 
```

Prism also has access to all WMT test sets via the
[sacrebleu](https://github.com/mjpost/sacrebleu) API. These can be
specified as arguments to `--src` and `--ref`, 
for a hypothetical system output $cand, as follows: 

```bash
./prism.py --cand $cand --ref sacrebleu:wmt19:de-en --model-dir m39v1/
```
which will cause it to use the English reference from the WMT19 German--English test set.
(Since the language is known, no `--lang` is needed).

To see all options, including segment-level scoring, run:

```bash
./prism.py -h
```

# Usage: Python Module

All functionality is also available in Python, for example:

```python
from prism import Prism
prism = Prism(model_dir='m39v1/', lang='en')

print('Prism identifier:', prism.identifier())

cand = ['Hi world.', 'This is a Test.']
ref = ['Hello world.', 'This is a test.']
src = ['Bonjour le monde.', "C'est un test."]

print('System-level metric:', prism.score(cand=cand, ref=ref))
print('Segment-level metric:', prism.score(cand=cand, ref=ref, segment_scores=True))
print('System-level QE-as-metric:', prism.score(cand=cand, src=src))
print('Segment-level QE-as-metric:', prism.score(cand=cand, src=src, segment_scores=True))
```

Which should produce:

>Prism identifier: {'version': '0.1', 'model': 'm39v1', 'seg_scores': 'avg_log_prob', 'sys_scores': 'avg_log_prob', 'log_base': 2}
>System-level metric: -1.0184666  
>Segment-level metric: [-1.4878583 -0.5490748]  
>System-level QE-as-metric: -1.8306842  
>Segment-level QE-as-metric: [-2.462842  -1.1985264]  

# Supported Languages

Albanian (sq), Arabic (ar), Bengali (bn), Bulgarian (bg), 
Catalan; Valencian (ca), Chinese (zh), Croatian (hr), Czech (cs), 
Danish (da), Dutch (nl), English (en), Esperanto (eo), Estonian (et),
Finnish (fi),  French (fr), German (de), Greek, Modern (el),
Hebrew (modern) (he),  Hungarian (hu), Indonesian (id), Italian (it),
Japanese (ja), Kazakh (kk), Latvian (lv), Lithuanian (lt), Macedonian (mk),
Norwegian (no), Polish (pl), Portuguese (pt), Romanian, Moldavan (ro),
Russian (ru), Serbian (sr), Slovak (sk), Slovene (sl), Spanish; Castilian (es),
Swedish (sv), Turkish (tr), Ukrainian (uk), Vietnamese (vi)

# Translating with the Multilingual NMT Model

If you wish to use the provided multilingual NMT model for translation, 
you will need to manually apply sentencpiece and 
force decode the output language tag (e.g., "\<fr\>") as the first token, as shown in the example below.

You will need to install [sentencepiece](https://github.com/google/sentencepiece).

Start with an example input file:
```bash
echo -e "Hi world.\nThis is a Test.\nSome of my Best Friends are Linguists." > data.src
```

Apply sentencepiece to the source file only:
```bash
spm_encode --model=m39v1/spm.model --output_format=piece < data.src > data.sp.src
```

Make a dummy target file 
with the same number of lines as the source file which contains
the desired a language code for each line.
Note that the language tags are in the model vocabulary 
and should not be split using sentencepiece:

```bash
LANG_CODE=fr
awk '{print "<'$LANG_CODE'>"}' data.sp.src > data.sp.tgt
```

Binarize the input data:

```bash
fairseq-preprocess --source-lang src --target-lang tgt \
    --tgtdict m39v1/dict.tgt.txt --srcdict m39v1/dict.src.txt \
    --testpref data.sp --destdir data_bin
```

Translate, force decoding the language tag to produce output in the desired language

```bash
fairseq-generate data_bin  --path m39v1/checkpoint.pt  --prefix-size 1 --remove-bpe sentencepiece
```

Which should produce output like:
>S-2     Some of my Best Friends are Linguists.  
>T-2     <fr>  
>H-2     -0.33040863275527954    <fr> Certains de mes meilleurs amis sont linguistes.  
>P-2     -2.1120 -0.5800 -0.1343 -0.1266 -0.1420 -0.1157 -0.0762 -0.0482 -0.1008 -0.5435 -0.0797 -0.1332 -0.1031  
>S-1     This is a Test.  
>T-1     <fr>  
>H-1     -0.5089232325553894     <fr> C'est un test.  
>P-1     -2.4594 -0.5629 -0.3363 -0.0851 -0.2041 -0.1533 -0.1690 -0.1013  
>S-0     Hi world.  
>T-0     <fr>  
>H-0     -1.2458715438842773     <fr> Le Monde.  
>P-0     -3.2018 -2.4925 -1.1116 -0.1039 -0.4667 -0.0987  

# Publications

If you the Prism metric and/or the provided multilingual NMT model, please cite:
```
@inproceedings{thompson-post-2020-automatic, 
    title={Automatic Machine Translation Evaluation in Many Languages via Zero-Shot Paraphrasing},
    author={Brian Thompson and Matt Post},
    year={2020},
    publisher = {arXiv preprint arXiv:2004.14564}
    url={https://arxiv.org/abs/2004.14564}
}
```
