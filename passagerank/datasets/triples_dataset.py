import csv

import torch
from transformers import AutoTokenizer

class TriplesDataset(torch.utils.data.IterableDataset):

    def __init__(
        self,
        triples_f,
        model_tag,
        batch_size,
        delimiter='\t',
        num_seen=None):
        super(TriplesDataset).__init__()

        self.triples_f = triples_f

        self.model_tag = model_tag

        self.batch_size = batch_size

        self.delimiter = delimiter

        # Set by worker_init_fn
        self.tokenizer = None

        self.num_seen = num_seen

    @staticmethod
    def worker_init_fn(worker_id):
        """ Initialize tokenizers post-fork """
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        dataset.load_tokenizer()

    def __iter__(self):

        if self.tokenizer is None:
            self.load_tokenizer()

        with open(self.triples_f, 'r') as infile:
            reader = csv.reader(infile, delimiter=self.delimiter)
            batch = {'queries': [], 'passages': [], 'labels': []}

            for i, row in enumerate(reader):
                
                if self.num_seen is not None and i < self.num_seen:
                    if i % (self.num_seen // 10) == 0:
                        print("Skipping: {}/{}".format(i, self.num_seen))
                    continue
                
                query, positive_passage, negative_passage = row
                batch['queries'].append(query)
                batch['passages'].append(positive_passage)
                batch['labels'].append(1)

                if len(batch['queries']) == self.batch_size:
                    
                    batch_inputs = self.tokenizer(
                        batch['queries'],
                        batch['passages'],
                        padding='longest',  # 'max_length'
                        truncation='longest_first',
                        return_tensors='pt'
                    )
                    batch_inputs['labels'] = torch.as_tensor(batch['labels'])
                    yield batch_inputs

                    batch = {'queries': [], 'passages': [], 'labels': []}

                batch['queries'].append(query)
                batch['passages'].append(negative_passage)
                batch['labels'].append(0)

                if len(batch['queries']) == self.batch_size:

                    batch_inputs = self.tokenizer(
                        batch['queries'],
                        batch['passages'],
                        padding='longest',  # 'max_length'
                        truncation='longest_first',
                        return_tensors='pt'
                    )
                    batch_inputs['labels'] = torch.as_tensor(batch['labels'])
                    yield batch_inputs

                    batch = {'queries': [], 'passages': [], 'labels': []}

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_tag)