
This repo includes the dataset extraction and preprocessing scripts.

We construct three datasets for outline-conditioned generation. We focus on fictitious generation, but also include the news domain for generalization.  We build on existing publicly available datasets for the  target  narratives,  paired  with  automatically constructed input outlines as described in detail in our paper. Here we provide the dataset ids to and the preprocessing scripts to construct the train/validation/test splits for experimentation.

# Prerequisites

numpy

Rake

nltk

TfidfVectorizer

# Pre-processing Data
## WikiPlots

#### 1. Steps for downloading
<a href="https://github.com/markriedl/WikiPlots">The Wikiplots corpus</a> consists of plots of movies, TV shows, and books scraped from Wikipedia.
Please use the scripts provided in the link to extract the dataset (we used Wikipedia from the 10/01/19 timestamp).  

You need to make changes to <a href="https://github.com/markriedl/WikiPlots/blob/22d975c92e1ac835a412ac001d95fb86d3d37960/wikiPlots.py#L81">line 81</a> of their code to replace '\n' with paragraph markers instead of with spaces:

`plot = plot.replace('\n\n', ' ##PM## ').replace('\r', '').strip()`

After processing is complete, you should replace the  '##PM##' markers with '&lt;p&gt;' before running the extract outlines script.

#### 2. Steps for extracting outlines
Run the <a href="./extract_outlines.py">extract_outlines.py</a> to extract the outline-labeled documents that can be used as input to the train Plotmachines fine-tuning models.

The output will provide you with a csv of the outlines and stories where each row is a paragraph from a story.  The columns are:
- story id: our format is "storyid_{int}" with the {int} after the underscore being this paragraph's index in the story (starting at 0)
- key/abstract: this is a binary signifier for us to know where the data came from, but it's just in "K" for every row, in wikiplots
- outline: the outline with points delimited by [SEP]
- discourse tag: I/B/C for intro, body, conclusion paragraphs respectively
- num_paragraphs: total number of paragraphs in this story
- paragraph: the paragraph text
- previous paragraph: text from the previous paragraph in the story

#### 3. Steps for splitting into train/dev/test splits
Please use the splits from <a href="./wikiplots_splits.txt">wikiplots_splits.txt</a> to construct the train, validation and text datasets that were used in the paper.  Note that some stories may be need to be removed (marked "flagged") due to potentially offensive and/or harmful content.

#### 4. Steps for removing offensive content: 
#### &nbsp; &nbsp; &nbsp;4a. Offensive story removal:

Some plots should be excluded from the data and are marked as 'flagged' instead of train/dev/test in the splits file.  These are stories that we have identified as coming from summaries of books/movies that are probably offensive.  We mostly try to remove stories that attack someone's identity (e.g. racist propaganda) or are problematically lurid. 

Details about how these stories were identified: we first used automatic toxicity models (using the Perspective API) to identify about stories that had toxicity scores above a manually chosen threshold (about 1500 stories).  We skimmed the automatically curated list of toxic stories and manually corrected the labels of about 200 of those stories that we believe were misclassified.  The remaining 1300 stories have been flagged in the data splits files.  There are limitations to this hybrid automatic/manual approach, and there may be some stories that were misclassified (either incorrectly labelled as inoffensive or incorrectly labelled as offensive).  We are continuing to prune the data to remove examples of offensive stories, so please let us know if you find any that we've missed.

####  &nbsp; &nbsp; &nbsp;4b. Offensive words:

Some stories may have instances of offensive words even though the overall story is not offensive.  Before training, you should check for swear words, slurs, etc.  We recommend adding a pre-processing step to replace these words with some special token.

####  &nbsp; &nbsp; &nbsp;4c. Additional precautions:

Even when taking these steps, there may be a few underlying themes in some stories that don't match modern values (for example, many older stories may express outdated views about gender roles).  There also may be stories containing sexual and violent content that - depending on the end use - may be inappropriate for a model to be trained on.  We therefore caution anyone using this data to be very careful in how they use models that are trained using these stories.  Please moderate output as necessary and appropriate for your end task.

## WritingPrompts

This is a story generation dataset, presented in <a href="https://arxiv.org/abs/1805.04833">Hierarchical Neural Story Generation, Fan et.al., 2018</a> collected from the /r/WritingPrompts subreddit - a forum where Reddit users compose short stories inspired by other users’ prompts. It contains over 300k human-written (prompt, story) pairs. We use the same train/dev/test split from the original dataset paper.  


## New York Times

NYTimes, <a href="https://catalog.ldc.upenn.edu/LDC2008T19">The New York Times Annotated Corpus LDC2008T19</a>, contains news articles. 
We use the scripts to parse NYT corpus and then split into train, validation and test using the list of keys provided in <a href="./nyt_splits.txt">nyt_splits.txt</a>

Lastly, run the <a href="./extract_outlines.py">extract_outlines.py</a> to extract the outline-labeled documents that can be used as input to the train Plotmachines fine-tuning models. This script can extract the wikiplots data. 


# After Pre-Processing 

Once the files are generated, rename the preprocessed files as "train_encoded.csv", "val_encoded.csv", "test_encoded.csv" and place them under "data_dir" folder, which is specificied as input parameter to the fine-truning script "train.py".

### Creating Pickle Files for h\_{i-1}

Our model expects there to be a &lowast;\_gpt.pkl or &lowast;\_gpt2.pkl file in the directory (depending on command line settings).  
In our paper, we talk about using an encoded representation of the previous paragraph (h\_{i-1}) which we computed using either gpt or gpt2 (depending on the PlotMachines settings).  To compute that, we used <a href="https://github.com/hrashkin/plotmachines/blob/65f3b4d79bdb7a14811c323becdc3fb78bdeb375/src/model/generate_stories.py#L17">this function here</a> which computes an average output embedding.  For training time, we precomputed h\_{i-1} from gold paragraphs and stored in pickle (pkl) files. 

For each row in the input csv files, there is an entry in the pickle file which should contain tuple of:
 - index in the csv data file (header row = 0)
 - string version of the previous paragraph (which has to match the last column from the input csv/jsonl files).
 - a vector representing the previous paragraph  

Please note: in order to match indices with the input files, there needs to be a "dummy" encoding at the beginning of the pickle file to line up with the header row of the input csv. That row will get ignored in the code.

