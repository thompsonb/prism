# Translating with the Prism Model

This README describes how to use the Prism model as a multilingual NMT model. You should have already installed the Prism environment and downloaded the Prism model -- see the [main README](../README.md). These instructions also assume you have installed [sentencepiece](https://github.com/google/sentencepiece).

You will need to manually apply sentencpiece and 
force decode the output language tag (e.g., "\<fr\>") as the first token,
as shown in the example below.

Start with an example input file:
```bash
echo -e "Hi world.\nThis is a Test.\nSome of my Best Friends are Linguists." > data.src
```

Apply sentencepiece to the source file only:
```bash
spm_encode --model=$MODEL_DIR/spm.model --output_format=piece < data.src > data.sp.src
```

Make a dummy target file 
with the same number of lines as the source file which contains
the desired language code for each line.
Note that the language tags are in the model vocabulary 
and should not be split using sentencepiece:

```bash
LANG_CODE=fr  # desired target language 
awk '{print "<'$LANG_CODE'>"}' data.sp.src > data.sp.tgt
```

Binarize the input data:

```bash
fairseq-preprocess --source-lang src --target-lang tgt \
    --tgtdict $MODEL_DIR/dict.tgt.txt --srcdict $MODEL_DIR/dict.src.txt \
    --testpref data.sp --destdir data_bin
```

Translate, force decoding the language tag to produce output in the desired language

```bash
fairseq-generate data_bin  --path $MODEL_DIR/checkpoint.pt  --prefix-size 1 --remove-bpe sentencepiece
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
If you use the Prism model, please cite the Prism paper from the [main README](../README.md#publications).