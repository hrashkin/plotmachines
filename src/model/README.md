# Instructions for training and running models:
### Prerequisites

pytorch

transformers

spacy

nltk

tqdm

rouge

shutil


### Training PlotMachines models
E.g.:
`python train.py --data_dir datadir --output_dir savedir --experiment_name pmfull --accum_iter 4 --n_batch 64 --p 90 --num_epochs 10 --use_model plotmachines --use_neighbor_feat --use_discourse`


Some important command line arguments:
  - ```use_neighbor_feat``` : use representation of previous paragraph in input (i.e neighboring features)
  - ```use_discourse``` : whether to use discourse type tags (`_i_`,`_b_`,`_c_`) or not
  - ```use_model={base/plotmachines}```: either the base gpt model without memory, or PlotMachines with memory
  - ```memstatesize={int}```: size of additional memory slots aside from the ones initialized from the outline (default:100)
  - ```n_batch={int}```: must be mulitple of number of gpus
  - ```output_dir```: a directory to save outputs to
  - ```data_dir```: location of all of the train/dev input files, each of which must be named {train/val/test}\_encoded.jsonl, should also contain {train/val/test}\_gpt.pkl files where the encoding of the previous paragraph is stored offline
  - ```p ={int}```: the % to use in nucleus sampling
  - ```repeattheta={float}```: how much to penalize repetitions. should be a float >= 1. (1=no penalty)


At the end of running the outputs are stored in output_dir/experiment_name/checkpoints:
  - `checkpoint_best.pt`: best checkpoint from training
  - `valeval.log` : a summary of rouge scores for single paragraph prediction post-training
  - `valeval.log.output.json` : generated outputs from paragraph prediction post-training

and output_dir/experiment_name/logs:
  - `output_losses.tsv` : log of loss scores on train/val examples throughout training
  - `output_rouge.tsv` : log of rouge scores on five val examples throughout training


### Generating stories

E.g.:
`python generate_stories.py --data_dir datadir --save_dir outputdir --n_batch 64 --p 90 --load_dir savedir/pmfull/checkpoints --use_model plotmachines --use_neighbor_feat --use_discourse`

Important command line arguments:
  - ```bodynum={int}```: number of body paragraphs to generate (default=3, for 5 paragraph format)
  - ```testset```: use test set instead of validation
  - ```use_model={base/plotmachines}```: either the base gpt model without memory, or PlotMachines with memory
  - ```memstatesize={int}```: size of additional memory slots aside from the ones initialized from the outline (default:100)
  - ```n_batch={int}```: must be mulitple of number of gpus
  - ```save_dir```: a directory to save generatins to
  - ```data_dir```: location of all of the train/dev input files, each of which must be named {train/val/test}\_encoded.jsonl (this script doesnt use the pkl files)
  - ```p ={int}```: the % to use in nucleus sampling
  - ```repeattheta={float}```: how much to penalize repetitions. should be a float >= 1. (1=no penalty)
  - ```load_dir={str}```: the location of checkpoint_best.pt saved from training

## Output format
At the end of running the generated story outputs are stored in `output_dir`:
  - `{val/test}eval.tsv`: generated stories
Note, each row is a single story paragraph and the paragraphs of each story might not be in contiguous order.
Each row contains:
```story-idx story-name  plot-outline paragraph-idx  paragraph-text```



### Additional Notes and Acknowledgements
Thanks to other codebases that were used in writing this code:
  - Huggingface's original gpt repo
  - Huggingface's current transformers repo
  - Transformer for abstractive summarization (used for parallel model classes): https://github.com/Andrew03/transformer-abstractive-summarization