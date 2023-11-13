import csv

from transformers import AutoTokenizer
import pandas as pd
import pyarrow.parquet as pq
import torch

class TrecValidationDataset(torch.utils.data.IterableDataset):

    def __init__(self, queries_f, query_passages_f, model_tag):
        super(TrecValidationDataset).__init__()

        # qid:passage:rank in RAM and grouped by qid
        self.pqfile = pq.ParquetFile(query_passages_f)

        # qid:query in RAM
        self.qid_to_query = load_queries_file(queries_f)

        self.model_tag = model_tag

        # Set by worker_init_fn
        self.tokenizer = None

        cpu = torch.device('cpu')
        gpu = torch.device('cuda')
        self.device = gpu if torch.cuda.is_available() else cpu

    @staticmethod
    def worker_init_fn(worker_id):
        """ Initialize tokenizers post-fork """
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        dataset.load_tokenizer()

    def __iter__(self):

        for batch in self.pqfile.iter_batches(
                        batch_size=100,
                        columns=['qid', 'pid', 'passage']):

            queries = [self.qid_to_query[qid] for qid in batch['qid'].tolist()]
            passages = batch['passage'].tolist()

            batch_inputs = self.tokenizer(
                queries,
                passages,
                padding='longest',  # 'max_length'
                truncation='longest_first',
                return_tensors='pt'
            )
            
            batch_inputs['qid'] = batch['qid'].tolist()
            batch_inputs['pid'] = batch['pid'].tolist()

            yield batch_inputs

    def __len__(self):
        return self.pqfile.metadata.num_rows // 100

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_tag)


def load_queries_file(queries_f):
    qid_to_query = {}
    with open(queries_f, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for qid, query in reader:
            qid_to_query[int(qid)] = query

    return qid_to_query