```
Installation cmd
pip3 install tf-keras

Ruuning TF original and calcualte perplexity
python tfopt.py facebook/opt-125m --dataset wikitext2 --seqlen 128

Running TF GPTQ 4bit model and calcualting perplexity
python optmodel.py facebook/opt-125m --dataset wikitext2 --wbits 4 --groupsize 128 --nsamples 8 --seqlen 128
```
