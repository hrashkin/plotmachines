## Dataset Construction for PlotMachines Experiments 

This repo includes the dataset extraction and preprocessing scripts.

We construct three datasets for outline-conditioned generation. We focus on fictitious generation, but also include the news domain for generalization.  We build on existing datasets for the  target  narratives,  paired  with  automatically constructed input outlines as described in detail in our paper. Here we provide the dataset ids to and the preprocessing scripts to construct the train/validation/test splits for experimentation.


### WikiPlots

<a href="https://github.com/markriedl/WikiPlots">The Wikiplots corpus</a> consists of plots of movies, TV shows, and books scraped from Wikipedia. 
Please use the scripts provided in the link to extract the dataset. Then use the script <a href="./wikiplots_splits.txt">wikiplots_splits.txt</a> to construct the train, validation and text datasets that were used in the paper.

