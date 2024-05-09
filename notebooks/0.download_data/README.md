# Downloading MitoCheck Dataset

This module focuses on downloading the MitoCheck dataset, a resource rich in data for profiling cellular division. The dataset's source is documented in [this article](https://www.nature.com/articles/nature08869). The research paper details the identification of hundreds of genes implicated in various biological functions such as cellular division, migration, and survival. Moreover, it reveals intermediate steps between major stages of cell division through the morphological profiles gathered in the study.

## Download Procedure

We've developed a shell script that converts the notebook into a script and then executes the Python script responsible for downloading the dataset. It's essential to note that the data we're obtaining comprises image-based profiles (quantified images), rather than the actual images themselves. These profiles encapsulate all the morphological information that will be used for training our deep learning model. The downloaded data will be stored within the `./data/raw` directory.

**To download the dataset type:**:

```bash
source download_data.sh
```
