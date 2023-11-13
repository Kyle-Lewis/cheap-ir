"""
Given an anserini run or validation set from TREC e.g., 
> head -n 10 2021_passage_top100.txt
> 2082 Q0 msmarco_passage_45_623131157 1 19.820700 Anserini
> 2082 Q0 msmarco_passage_10_669481249 2 18.954300 Anserini
> 2082 Q0 msmarco_passage_10_669572296 3 18.954298 Anserini
> ...

Create *2021_passage_top100_hydrated.parquet*:
> qid (int) | pid (str)                       | passage (str)
> 2082      | "msmarco_passage_45_623131157"  | "Some passage text here..."
> 2082      | "msmarco_passage_10_669481249"  | "Some passage text there..."
> 2082      | "msmarco_passage_10_669572296"  | "Some other text here..."

The resulting file can then be used as input to the model to generate runs for
the official trec_eval tool without extracting passages from the corpus for
tokenization.
"""

from datetime import timedelta
from pathlib import Path
import argparse
import gzip
import json
import time
import os

import pandas as pd

def block_passages(corpus_dir, bundle_num, offsets):
    """ More efficiently gather sorted passages from one bundle file
    Offsets should be unique & sorted
    
    Yields
    """

    bundle_name = 'msmarco_passage_{}.gz'.format(bundle_num)
    gzip_file = corpus_dir.joinpath(bundle_name)
    
    curr = 0
    with gzip.open(gzip_file, 'rb') as infile:
        
        for off in offsets:
            seekbytes = off - curr
            infile.seek(seekbytes, 1)
            passage_data = json.loads(infile.readline())
            curr = infile.tell()

            yield passage_data['passage']


def get_passage_map(corpus_dir, block_num, block_pids):
    """ block_pids should be unique & sorted for good performance """
    
    passage_map = {}
    offsets = [int(p.split('_')[-1]) for p in block_pids]

    passages = block_passages(corpus_dir, block_num, offsets)
    for pid, passage in zip(block_pids, passages):
        passage_map[pid] = passage
    
    return passage_map


def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    # required
    parser.add_argument(
        "--corpus-dir", type=Path, required=True,
        help="Folder containing the 70 MSMARCOv2 passage corpus .gz files"
    )

    parser.add_argument(
        "--candidates-file", type=Path, required=True,
        help="Input validation candidates from TREC DL track e.g., /2021_passage_top100.txt"
    )
    args = parser.parse_args()

    # resolve relative paths and validate
    if not os.path.isabs(args.corpus_dir):
        args.corpus_dir = \
            Path(__file__).absolute().parent.joinpath(args.corpus_dir)
    
    if not os.path.isabs(args.candidates_file):
        args.candidates_file = \
            Path(__file__).absolute().parent.joinpath(args.candidates_file)

    if not (args.corpus_dir.exists() and args.corpus_dir.is_dir()):
            raise argparse.ArgumentTypeError(
                "Invalid corpus directory:", args.corpus_dir)
                
    if not (args.candidates_file.exists() and args.candidates_file.is_file()):
        raise argparse.ArgumentTypeError(
            "Invalid candidates file:", args.candidates_file)


    for arg, value in args.__dict__.items():
        print("{:>30} - {}".format(arg, value))

    ## done with arguments

    cols = ['qid', 'Q0', 'pid', 'rank', 'score', 'runstring']
    usecols = ['qid', 'pid']
    query_passage_blocks = pd.read_csv(
        args.candidates_file, names=cols, usecols=usecols, sep=' ')

    print('\n', query_passage_blocks.head(), '\n')

    # in "msmarco_passage_##_123456789" refers to the gzipped jsonl file "block"
    # that the passage will be found in. Gather this in a column so we can operate on
    # one block file at a time
    query_passage_blocks['block_num'] = \
    query_passage_blocks['pid'].map(lambda s: s.split('_')[2])

    # For each group of passage ids (grouped by the .gz file segment they came from)
    # gather the passage texts by hopping through the sorted offsets, and map the
    # results to a new column 'passage'.
    start_time = time.time()
    for block_num, block in query_passage_blocks.groupby('block_num'):

        elapsed = str(timedelta(seconds=int(time.time() - start_time)))
        print("extracting: ({}/70) - elapsed: {}".format(block_num, elapsed), end="\r")
        
        # dont spend time loading dupes
        block_pids = block['pid'].values.tolist()
        block_pids = sorted(list(set(block_pids)), key=lambda s: int(s.split('_')[-1]))

        # update the 'passage' column but only for the current block
        pid_to_passage = get_passage_map(args.corpus_dir, block_num, block_pids)
        mask = (query_passage_blocks['block_num'] == block_num)
        to_map = query_passage_blocks[mask]

        query_passage_blocks.loc[mask, 'passage'] = \
            to_map['pid'].map(pid_to_passage, na_action='ignore')

    # validate - all pids have been mapped to a non-null passage string
    assert not query_passage_blocks['passage'].isnull().values.any()

    # validate - no passage string is empty
    assert query_passage_blocks['passage'].str.len().values.all()

    print('\n', query_passage_blocks.head(), '\n')


if __name__ == "__main__":
    main()