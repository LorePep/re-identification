# re-identification
Code for re-identification from images. This code demonstrate how to use Triple Loss and online hard batch mining
to solve the re-identification problem.

For more info refer to [this blog post](https://lorenzopeppoloni.com/reidentification/).

The code uses the dataset from the  [Humpback Whale Identification Kaggle competition](https://www.kaggle.com/c/humpback-whale-identification). Bounding boxes are also included ([as computed by Martin Piotte](https://www.kaggle.com/martinpiotte/bounding-box-model/output) during the competition).

## Installation


## How to run

```bash
python train.py --train-path [path-to-train-folder] --labels [path-to-labels-csv] --boxes [path-to-boxes-csv] --output-dir [output-dir] [--verbose] [--hard]
```

If specifying `--hard` the train will be performed using hard batch mining.
