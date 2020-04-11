# Instructions for training and running models:
## 0) Prereqs:
- need:
  ```pytorch, pytorch-transformers, spacy, nltk, pyrouge, tqdm, ftfy```
- add the gpt model (https://github.com/openai/finetune-transformer-lm/tree/master/model) as a model subdirectory in this folder

## 1) Train a new model using the train_full_decoder.py script.

E.g.:
`python train_full_decoder.py --data_dir datadir --output_dir savedir --experiment_name modelv1 --accum_iter 4 --n_batch 64 --p 60 --num_epochs 10 --use_model full `


Important command line arguments:
  - ```--exclude_kw``` : *don't* use keywords as input (i.e. an unconditional lm set-up)
  - ```--use_neighbor_feat``` : use BERT clf representation of previous paragraph in input (i.e neighboring features)
  - ```--use_aux_loss``` : use auxillary losses - minimize distance of current paragraph from Bert rep, and similarity from Bert rep of previous paragraph
  - ```use_model={vanilla/full}```: either whether to use discourse type tags (`_i_`,`_b_`,`_c_`) or not
  - ```n_batch={int}```: must be mulitple of number of gpus

At the end of running the outputs are stored in output_dir/experiment_name/checkpoints:
  - `checkpoint_best.pt`: best checkpoint from training
  - `valeval.log` : a summary of rouge scores for single paragraph prediction post-training
  - `valeval.log.output.json` : generated outputs from paragraph prediction post-training

and output_dir/experiment_name/logs:
  - `modeldescr.txt`: model description
  - `output_losses.tsv` : log of loss scores on train/val examples throughout training
  - `output_rouge.tsv` : log of rouge scores on five val examples throughout training

## 2) Evaluate on single paragraph predictions.

E.g.:
```- python evaluate.py --data_dir datadir --save_dir outputdir --n_batch 64 --p 60 --use_model full --load_dir savedir/modelv1/checkpoint```

Important command line arguments:
  - ```--testset```: use test set instead of validation
  - ```--p ={int}```: the % to use in nucleus sampling
  - ```--load_dir={str}```: the location of checkpoint_best.pt saved from training
  - ```--exclude_kw``` : *don't* use keywords as input (i.e. an unconditional lm set-up)
  - ```--use_neighbor_feat``` : use BERT clf representation of previous paragraph in input (i.e neighboring features)
  - ```--use_aux_loss``` : use auxillary losses - minimize distance of current paragraph from Bert rep, and similarity from Bert rep of previous paragraph
  - ```use_model={vanilla/full}```: either whether to use discourse type tags (`_i_`,`_b_`,`_c_`) or not
  - ```n_batch={int}```: must be mulitple of number of gpus

At the end of running the outputs are stored in `output_dir`:
  - `valLOSS.txt`: avg loss of each batch
  - `valROUGE.json.output.json` : generated outputs from single paragraph prediction
  - `valROUGE.json` : average rouge from generated outputs from single paragraph prediction

## 3) Generate full documents from keywords.

E.g.:
```- python generate_full_doc.py --data_dir datadir --save_dir outputdir --n_batch 64 --p 60 --use_model full --load_dir savedir/modelv1/checkpoint```

Important command line arguments:
  - ```--bodynum={int}```: number of body paragraphs to generate (default=3, of 5 paragraph format)
  - ```--testset```: use test set instead of validation
  - ```--p ={int}```: the % to use in nucleus sampling
  - ```--load_dir={str}```: the location of checkpoint_best.pt saved from training
  - ```--exclude_kw``` : *don't* use keywords as input (i.e. an unconditional lm set-up)
  - ```--use_neighbor_feat``` : use BERT clf representation of previous paragraph in input (i.e neighboring features)
  - ```--use_aux_loss``` : use auxillary losses - minimize distance of current paragraph from Bert rep, and similarity from Bert rep of previous paragraph
  - ```use_model={vanilla/full}```: either whether to use discourse type tags (`_i_`,`_b_`,`_c_`) or not
  - ```n_batch={int}```: must be mulitple of number of gpus

At the end of running the outputs are stored in `output_dir`:
  - `valeval.tsv`: generated documents