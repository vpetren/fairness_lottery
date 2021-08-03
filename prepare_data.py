import os
import ast
import sys
import csv
import argparse
import json
import random
import datetime
import numpy as np
import pandas as pd
from pprint import pprint
from collections import Counter, defaultdict

import torchtext
import torch
from torchtext import data
from torchtext import datasets
from torchtext.data import BucketIterator, Dataset, Example
from torchtext.vocab import Vocab

from nltk import word_tokenize
from math import ceil


random.seed(121)


def load_data(fn):
    return [ast.literal_eval(line) for line in open(fn)]


def flatten_reviews(users):
    """ User entry can contain multiple reviews,
        this function turn each review into seperate data points.
    """
    reviews = []
    for user in users:
        n_r = len(user['reviews'])
        for review in user['reviews']:
            rev = user.copy()
            rev['reviews'] = review
            reviews.append(rev)

    return reviews


def extract_relevant_info(d, country=None):
    """ Extract relevant information and make sure missing values are
        replaced with default values.
    """
    date = get_datetime(d['reviews']['date'])
    return {'rating': simplify_rating(int(d['reviews']['rating'])),
            'text': ' '.join(d['reviews']['text']),
            'gender': d['gender'] if 'gender' in d else None,
            'location': get_location(d, country),
            'age': date.year - int(d['birth_year']) if 'birth_year' in d else -1,
            'date': date,
            }


def get_datetime(date):
    """ date input format YYYY-MM-DDTHH:MM:SS  """
    if date is None:
        return None
    date, time = date.split('T')
    year, month, day = date.split('-')
    return datetime.datetime(int(year), int(month), int(day))


def get_location(d, country):
    loc = None
    if country == 'denmark' and 'NUTS-2' in d:
        loc = d['NUTS-2']
    elif country in ['united_kingdom', 'germany', 'france'] and 'NUTS-1' in d:
        loc = d['NUTS-1']
    return loc


def simplify_rating(rating):
    if rating > 3:
        return 1
    if rating < 3:
        return 0
    raise Exception()


def load_and_process_data(fn, country=None):
    data = load_data(fn)
    data = flatten_reviews(data)

    data_processed = []
    errs = 0
    for d in data:
        try:
            processed = extract_relevant_info(d, country=country)
            if len(processed['text'].strip()) < 1:
                errs += 1
                continue
            data_processed.append(processed)
        except Exception as e:
            errs += 1
            continue
    return data_processed


def save_split(fn, data):
    with open(fn, 'w') as f:
        for line in data:
            json.dump(line, f)
            f.write('\n')


def load_test_iterator(fname, text_field, label_field, bsz, device):
    d = data.TabularDataset(path=fname, format='tsv', fields=[('text', text_field), ('label', label_field)])
    iterator = data.BucketIterator(d, batch_size=bsz, sort_key=lambda x: len(x.text),
                                   sort_within_batch=True, device=device, train=False)

    return (fname.split('/')[-1].split('_test.tsv')[0], iterator)


def save_tsv(data, path, country, split):
    with open(path + country + '_' + split + '.tsv', 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        [tsv_writer.writerow([d['text'], d['age'], d['gender'], d['location'], d['rating']]) for d in data]


def load_demographic_torchtext(datadir, batch_size, device):
    print('Loading data...')
    TEXT = torchtext.data.Field(tokenize=word_tokenize, include_lengths=True,
                                lower=True, batch_first=True)
    LABEL = torchtext.data.LabelField(dtype=torch.long, batch_first=True)

    # Create your dataset from examples
    dsets_fns = defaultdict(list)
    dir_fns = [f"{datadir}/{f}" for f in os.listdir(datadir) if os.path.isfile(os.path.join(datadir, f))]
    for fn in dir_fns:
        if 'train' in fn:
            dsets_fns['train'] = fn
        elif 'test' in fn:
            dsets_fns['test'].append(fn)

    # Get train/val split from main dataset
    train_val = data.TabularDataset(path=dsets_fns['train'], format='tsv', fields=[('text', TEXT), ('label', LABEL)])
    train_data, val_data = train_val.split(split_ratio=[0.8, 0.2], random_state=random.getstate())

    # Create your dataset vocabulary
    print('Building vocabs')
    MAX_VOCAB_SIZE = 25000
    TEXT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE, min_freq=3)
    LABEL.build_vocab(train_data)

    train_iterator, val_iterator = data.BucketIterator.splits((train_data, val_data),
                                                              batch_size=batch_size,
                                                              sort_key=lambda x: len(x.text),
                                                              sort_within_batch=True,
                                                              device=device)

    test_iterators = []
    for test in dsets_fns['test']:
        test_iterators.append(load_test_iterator(test, TEXT, LABEL, batch_size, device))

    return train_iterator, val_iterator, test_iterators, TEXT.vocab, LABEL.vocab


def imdb_data():
    SEED = 1234
    torch.manual_seed(SEED)

    TEXT = data.Field(tokenize='spacy', include_lengths=True, batch_first=True)
    LABEL = data.LabelField(dtype=torch.long, batch_first=True)

    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

    train_data, valid_data = train_data.split(random_state=random.seed(SEED))

    MAX_VOCAB_SIZE = 25000

    TEXT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE)
    LABEL.build_vocab(train_data)

    BATCH_SIZE = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        device=device,
        repeat=True
    )
    return train_iterator, valid_iterator, test_iterator, TEXT.vocab, LABEL.vocab


def to_tsv(fname, splits=[0.8, 0.1, 0.1]):
    data = load_and_process_data(fname)

    # Prepare splits
    country = fname.split('/')[-1].split('.')[0]
    inds = list(range(len(data)))
    n_train = ceil(len(inds) * splits[0])
    n_dev = ceil(len(inds) * splits[1])

    random.shuffle(inds)

    train_inds = inds[:n_train]
    dev_inds = inds[n_train:n_train + n_dev]
    test_inds = inds[n_train + n_dev:]

    train = [data[i] for i in train_inds]
    dev = [data[i] for i in dev_inds]
    test = [data[i] for i in test_inds]

    save_path = '/'.join(fname.split('/')[:-1]) + '/' + country + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_tsv(train, save_path, country, 'train')
    save_tsv(dev, save_path, country, 'dev')
    save_tsv(test, save_path, country, 'test')


def get_date_thresh(demographics, t):
    min_date = datetime.datetime(2020, 1, 1)
    for demo in demographics.keys():
        dates = sorted([r['date'] for r in demographics[demo]])
        threshold = dates[::-1][t]
        min_date = min(min_date, threshold)
    return min_date


def get_demographics(reviews, country):
    demographics = defaultdict(list)
    for r in reviews:
        if r['age'] > 0 and r['location'] is not None and r['gender'] is not None and r['date'] is not None:
            if (country == 'denmark' and 'DK' not in r['location']) or \
               (country == 'germany' and 'DE' not in r['location']):
                continue

            a = 'old' if r['age'] > 35 else 'young'
            triple = (a, r['gender'], r['location'])
            demographics[triple].append(r)
        else:
            demographics['other'].append(r)

    #for k, v in demographics.items():
    #    print(k, len(v))

    return demographics


def split_by_date(demographics, threshold, n_samples):
    demographics_splits = defaultdict(list)
    for demo in demographics.keys():
        before, after = [], []

        for review in demographics[demo]:
            if review['date'] < threshold:
                before.append(review)
            else:
                after.append(review)

        after = np.random.choice(after, n_samples, replace=False)

        string = "other" if "other" in demo else f"{'_'.join(demo)}"

        demographics_splits[f"{string}_test"] = after
        demographics_splits["train"].extend(before)

    return demographics_splits


def split_by_random_sample(demographics, n_samples, cc=True):
    demographics_splits = defaultdict(list)
    for demo in demographics.keys():
        reviews = demographics[demo]

        if len(reviews) < n_samples:
            print(f"{demo} contains too few samples ({len(reviews)}), need atleast {n_samples}")
            continue

        idxs = [i for i in range(len(reviews))]
        random.shuffle(idxs)
        test_idxs = idxs[:n_samples]
        train_idxs = idxs[n_samples:]

        test = [reviews[i] for i in test_idxs]
        train = [reviews[i] for i in train_idxs]

        string = "other" if "other" in demo else f"{'_'.join(demo)}"

        demographics_splits[f"{string}_test"] = test
        demographics_splits["train"].extend(train)

    return demographics_splits


def balance_ratings(reviews):
    """ Downsample majority classes to match minority class """
    ratings_sort = defaultdict(list)
    for r in reviews:
        ratings_sort[r['rating']].append(r)

    n_min_class = min([len(v) for k, v in ratings_sort.items()])

    rebalanced_reviews = []
    for rating, revs in ratings_sort.items():
        rebalanced_reviews.extend(np.random.choice(revs, n_min_class, replace=False))

    random.shuffle(rebalanced_reviews)

    return rebalanced_reviews


def save_demo_tsv(data, path, country, split):
    with open(path + country + '_' + split + '.tsv', 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        [tsv_writer.writerow([d['text'], d['rating']]) for d in data]


def to_demographic_tsv(fname, t=500, balanced=False, split_random=False):
    country = fname.split('/')[-1].split('.')[0]

    data = load_and_process_data(fname, country=country)
    demographics = get_demographics(data, country)
    #overview = []
    #for d in demographics.keys():
    #    overview.append([d, len(demographics[d])])
    #overview.sort(key=lambda x: x[1], reverse=True)
    #pprint(overview)
    #input()
    if not split_random:
        min_thresh = get_date_thresh(demographics, t)
        demographics_splits = split_by_date(demographics, min_thresh, t)
    else:
        demographics_splits = split_by_random_sample(demographics, t)

    if balanced:
        demographics_splits['train'] = balance_ratings(demographics_splits['train'])

    save_path = '/'.join(fname.split('/')[:-1]) + '/' + country + '_demographics/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for d in demographics_splits.keys():
        #print(d, len(demographics_splits[d]))
        save_demo_tsv(demographics_splits[d], save_path, country, d)


def get_cc_demographics(comments, demos=['gender', 'race']):
    # Available subdemographics
    sub_demos = {
        'race': ['black', 'white', 'asian', 'latino', 'other_race_or_ethnicity'],
        'gender': ['male', 'female', 'transgender', 'other_gender'],
        'religion': ['christian', 'jewish', 'muslim', 'hindu', 'buddhist', 'atheist', 'other_religion'],
        'orientation': ['heterosexual', 'homosexual_gay_or_lesbian', 'bisexual', 'other_sexual_orientation'],
        'disability': ['physical_disability', 'intellectual_or_learning_disability', 'psychiatric_or_mental_illness', 'other_disability']
    }
    demographics = defaultdict(list)

    # Binarize toxicity
    comments['toxicity'] = np.where(comments['toxicity'] >= 0.5, 1, 0)

    # Separate comments with and without all specified demographics presents
    demo_ = (comments[f'na_{demos[0]}'] == 0) & (comments[f'na_{demos[1]}'] == 0)
    na_demo_rows = comments.loc[-demo_]
    demo_rows = comments.loc[demo_]

    demographics['other'] = na_demo_rows[['comment_text', 'toxicity']].values

    # Divide into demographics
    for i, row in demo_rows.iterrows():
        d1 = row[sub_demos[demos[0]]].astype('float64').idxmax(axis=1)
        d2 = row[sub_demos[demos[1]]].astype('float64').idxmax(axis=1)
        #import pdb; pdb.set_trace()
        comment_label = row[['comment_text', 'toxicity']].values
        demographics[(d1, d2)].append(comment_label)

    #for k, v in demographics.items():
    #    print(k, len(v))

    return demographics


def save_cc_tsv(data, path, split):
    with open(path + split + '.tsv', 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        [tsv_writer.writerow([d[0], d[1]]) for d in data]


def to_civil_comments_tsv(fname, t=500):
    data = pd.read_csv(fname, delimiter=',')
    demographics = get_cc_demographics(data, demos=['gender', 'race'])

    demographics_splits = split_by_random_sample(demographics, t)

    save_path = '/'.join(fname.split('/')[:-1]) + '/' + 'civil_comments_demographics/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for d in demographics_splits.keys():
        print(d, len(demographics_splits[d]))
        save_cc_tsv(demographics_splits[d], save_path, d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='Name of trustpilot file to preprocess')
    parser.add_argument('--mode', choices=['tsv', 'demographic', 'cc'], default='tsv')
    parser.add_argument('--balanced', action="store_true")
    parser.add_argument('--split_rand', action="store_true")
    parser.add_argument('--threshold', type=int, default=500)

    args = parser.parse_args()

    if args.mode == 'tsv':
        to_tsv(args.data)
    elif args.mode == 'demographic':
        to_demographic_tsv(args.data, args.threshold, balanced=args.balanced, split_random=args.split_rand)
    elif args.mode == 'cc':
        to_civil_comments_tsv(args.data, args.threshold)
