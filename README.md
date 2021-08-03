# Code for the paper "Is the Lottery Fair?: Evaluating Winning Tickets Across Demographics"
To appear in Findings of ACL 2021

# Dependencies
To install the python dependencies in a virutal environment:
```
pip install -r requirements.txt
```

The code was originally run using Python 3.6

# Download and prepocess the data
To download the Trustpilot Corpus, run the `dl_trustpilot.sh` script and process with:
```
python prepare_data.py --data data/trustpilot/denmark.auto-adjusted_gender.NUTS-regions.jsonl.tmp --mode demographic --balanced --split_rand --threshold 500
python prepare_data.py --data data/trustpilot/united_kingdom.auto-adjusted_gender.NUTS-regions.jsonl.tmp --mode demographic --balanced --split_rand --threshold 200
python prepare_data.py --data data/trustpilot/germany.auto-adjusted_gender.NUTS-regions.jsonl.tmp --mode demographic --balanced --split_rand --threshold 100
```
To download and process CivilComments dataset, first get the data `all_data.csv` from [Kaggle](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data) and place it in the `data/civil_comments` directory, then run the following scripts:
```
python augment_identities_and_split.py --root data/civil_comments
python prepare_data.py --data data/civil_comments/all_data_with_identities.csv --mode cc --threshold 100
```

# Running pruning experiments
An example run script has been provided in `run.sh` for the German part of the Trustpilot corpus:
```
python train_mlp.py --data data/trustpilot/germany_demographics \
                    --hidden-dim 64 \
                    --embed-dim 32 \
                    --num-epochs 20 \
                    --batch-size 8 \
                    --pruning_pct 0.35 \
                    --model mlp \
                    --num-pruning-iter 20 \
                    --bidirectional \
                    --pruning_method magnitude \
                    --pruning_structure layerwise \
                    #--dro
```

In the `run_slurm/` directory you can find the scripts used to produce the results from the paper, you'd probably want to customize them for your own system.

# Citation
If you found this repository useful for your own publication, please cite the original paper:
```
@inproceedings{hansen-sogaard-2021-lottery,
    title = "Is the Lottery Fair? Evaluating Winning Tickets Across Demographics",
    author = "\textbf{Hansen}, \textbf{Victor Petr{\'e}n Bach}  and
      S{\o}gaard, Anders",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.284",
    doi = "10.18653/v1/2021.findings-acl.284",
    pages = "3214--3224",
}
```

# TODOs
* Plotting script