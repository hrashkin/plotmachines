
This repo includes the dataset extraction and preprocessing scripts.

We construct three datasets for outline-conditioned generation. We focus on fictitious generation, but also include the news domain for generalization.  We build on existing publicly available datasets for the  target  narratives,  paired  with  automatically constructed input outlines as described in detail in our paper. Here we provide the dataset ids to and the preprocessing scripts to construct the train/validation/test splits for experimentation.

### Prerequisites

numpy

Rake

nltk

TfidfVectorizer

### WikiPlots

<a href="https://github.com/markriedl/WikiPlots">The Wikiplots corpus</a> consists of plots of movies, TV shows, and books scraped from Wikipedia.
Please use the scripts provided in the link to extract the dataset.  You need to make changes to <a href="https://github.com/markriedl/WikiPlots/blob/22d975c92e1ac835a412ac001d95fb86d3d37960/wikiPlots.py#L81">line 81</a> of their code to replace '\n' with'&lt;p&gt;' paragraph markers instead of with spaces. Then use the splits from <a href="./wikiplots_splits.txt">wikiplots_splits.txt</a> to construct the train, validation and text datasets that were used in the paper.

Note: Some plots should be excluded from the data and are marked as 'flagged' instead of train/dev/test in the splits file.  These are stories that we have identified as offensive content.  We are continuing to prune the data to remove examples of these stories, so please let us know if you find stories that you think should be removed.

### WritingPrompts

This is a story generation dataset, presented in <a href="https://arxiv.org/abs/1805.04833">Hierarchical Neural Story Generation, Fan et.al., 2018</a> collected from the /r/WritingPrompts subreddit - a forum where Reddit users compose short stories inspired by other users’ prompts. It contains over 300k human-written (prompt, story) pairs. We use the same train/dev/test split from the original dataset paper.  


### New York Times

NYTimes, <a href="https://catalog.ldc.upenn.edu/LDC2008T19">The New York Times Annotated Corpus LDC2008T19</a>, contains news articles. 
We use the scripts to parse NYT corpus and then split into train, validation and test using the list of keys provided in <a href="./nyt_splits.txt">nyt_splits.txt</a>

Lastly, run the <a href="./extract_outlines.py">extract_outlines.py</a> to extract the outline-labeled documents that can be used as input to the train Plotmachines fine-tuning models. This script can extract the wikiplots data. 


### After Pre-Processing 

Once the files are generated, rename the preprocessed files as "train_encoded.csv", "val_encoded.csv", "test_encoded.csv" and place them under "data_dir" folder, which is specificied as input parameter to the fine-truning script "train.py".

### Creating Pickle Files

Our model expects there to be a &lowast;\_gpt.pkl or &lowast;\_gpt2.pkl file in the directory (depending on command line settings).  
In our paper, we talk about using an encoded representation of the previous paragraph (h\_{i-1}) which we computed using either gpt or gpt2 (depending on the PlotMachines settings).  To compute that, we used <a href="https://github.com/hrashkin/plotmachines/blob/65f3b4d79bdb7a14811c323becdc3fb78bdeb375/src/model/generate_stories.py#L17">this function here</a> which computes an average output embedding.  For training time, we precomputed h\_{i-1} from gold paragraphs and stored in pickle (pkl) files. 

For each row in the input csv files, there is an entry in the pickle file which should contain tuple of:
 - index in the csv data file (header row = 0)
 - string version of the previous paragraph (which has to match the last column from the input csv/jsonl files).
 - a vector representing the previous paragraph  
 
Please note: in order to match indices with the input files, there needs to be a "dummy" encoding at the beginning of the pickle file to line up with the header row of the input csv. That row will get ignored in the code.

