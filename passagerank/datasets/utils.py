from torch.utils.data import DataLoader

from passagerank.datasets.triples_dataset import (
    TriplesDataset,
    TriplesDatasetForDPR
)

from passagerank.datasets.trec_validation_dataset import (
    TrecValidationDataset
)

def get_triples_dataloader(
    path: str,
    model_tag: str,
    batch_size: int,
    offset: int = 0,
) -> (TriplesDataset, DataLoader):
    
    dataset = TriplesDataset(
        path,
        model_tag=model_tag,
        batch_size=batch_size,
        offset=offset,
    )

    dataloader = DataLoader(
        dataset,
        num_workers=1,
        worker_init_fn=TriplesDataset.worker_init_fn,
        batch_size=None
    )

    return dataset, dataloader


def get_triples_dpr_dataloader(
    path: str,
    model_tag: str,
    batch_size: int,
    offset: int = 0,
) -> (TriplesDatasetForDPR, DataLoader):
    
    dataset = TriplesDatasetForDPR(
        path,
        model_tag=model_tag,
        batch_size=batch_size,
        offset=offset,
    )

    dataloader = DataLoader(
        dataset,
        num_workers=1,
        worker_init_fn=TriplesDatasetForDPR.worker_init_fn,
        batch_size=None
    )

    return dataset, dataloader


def get_trec_validation_dataloader(
    eval_queries_file: str,
    eval_candidates_file: str,
    model_tag: str,
) -> (TrecValidationDataset, DataLoader):
    
    dataset = TrecValidationDataset(
        queries_f=eval_queries_file,
        query_passages_f=eval_candidates_file,
        model_tag=model_tag
    )

    dataloader = DataLoader(
        dataset,
        num_workers=1,
        worker_init_fn=TrecValidationDataset.worker_init_fn,
        batch_size=None
    )

    return dataset, dataloader