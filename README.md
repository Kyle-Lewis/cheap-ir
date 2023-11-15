# Introduction

My goals with this project are to explore the limits of applying pruning and compression to re-ranking models, and to information retrieval (IR) models generally. In a trial run with a small model, I achieved a 70% reduction in inference time while maintaining 90% of model performance.

Information retrieval is the task of finding relevant documents from a large collection in response to a user query. In many IR systems, a large number (e.g., 1000) of candidate documents are first retrieved by a fast model that can search the entire corpus in a reasonable amount of time. These candidates are then *re-ranked* in order of their relevance as determined by a more sophisticated model. If the re-ranking model performs well, the system can provide only the top results (e.g., 10) to the user while remaining confident that relevant information has been returned. The two phase strategy aims to strike a balance between performance and response time - to search the entire corpus with the more sophisticated model alone would make the system unusably slow.

Model pruning encompasses a variety of techniques for removing the weights and connections from neural networks while maintaining as much of their accuracy as possible. Pruning is typically applied incrementally and in tandem with continued training of the model: After elements of the model are selected for removal, the remaining model receives additional training to recover and maintain its predictive ability.

In the context of IR, pruning the larger re-ranking models offers a number of benefits. First, a pruned model will simply be cheaper to host, and the provider will be able to scale their services more economically - the savings coming from reduced compute costs and/or smaller memory requirements, depending on the particular pruning technique and model. The re-ranker will also be faster. With a faster re-ranker, the provider can pass on faster response times to end users, but they can also choose to "use" this time by increasing the number of candidates which are considered by the re-ranker while still maintaining satisfactory response times. With the additional candidates, the system will perform better: If a good answer to a question is the 1001st candidate, then increasing the candidates beyond 1000 gives the re-ranker the opportunity to return that candidate in the top results.

Two-phase IR systems use various techniques when retrieving candidate documents in the first stage. It is worth distinguishing between two major classes: "traditional" ranking techniques like [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) and "dense retrieval" techniques. In its basic form, the traditional BM25 technique is a weighted keyword search and historically has been the basis for a Lucene search index. In dense retrieval, documents are instead encoded as dense vectors using language models. Upon receiving a query, the vector representation of that query is compared with document vectors, and the nearest neighbors are taken as the candidates. Dense retrieval comes with the downside of a slower and more expensive up front cost to index a new corpus. One of the primary benefits of dense retrieval systems is that the second stage re-ranker can be made [incredibly fast](https://arxiv.org/abs/2004.12832) by using the pre-computed document vectors and only computing the representation for the query on the fly, where the query is typically very small compared to a document.

In practice, the traditional and dense retrieval techniques get different things right when it comes to finding good candidate documents, and so many systems choose to retrieve some candidates from both.

I applied the techniques described below to re-ranking models with a **traditional** candidate ranking technique (a BM25 index built using the [Anserini toolkit](https://github.com/castorini/anserini)).

# Results Overview

In order to iterate quickly with my initial experiments I have trialed my pruning methods using the smaller distilbert base model.

| Model           | NDCG@10     | inference time (CPU) (ms)   | inference time (GPU) (ms)  | FLOPs (est.)
| -----------     | ----------- | -----------                 | -----------                | ------------
| distilbert-base | 0.6018      | 14163 +/- 129               | 1827 +/- 17                | 2.42T       
| distilbert[A]   | 0.5399      | 8788 +/- 131                | 1502 +/- 10                | 1.75T       
| distilbert[B]   | 0.5827      | 10287 +/- 317               | 822 +/- 17                 | 1.33T       
| distilbert[C]   | 0.5382      | 5508 +/- 347                | 499 +/- 21                 | 0.74T       

[See below](#full-results-table) for a more detailed results table.

- Timings were taken for re-ranking to a depth of 100 candidates.
- CPU timings were taken on a 6 core, 3.70GHz CPU.
- GPU timings were taken on a 1080ti.
- Floating point operations (FLOPs), were calculated using [fvcore](https://github.com/facebookresearch/fvcore)

Timings and FLOPs were taken using synthetic worst case data (full context window every time). Real life performance would depend on the passage segmentation strategy for the corpus.

I'm measuring ranking quality using [Normalized Discounted Cumulative Gain](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) (NDCG). In short, NDCG measures the quality of ranked results for a given query accounting for:
- Some relevant documents are *more* relevant than others (labels are on a graded scale from 1-4)
- It would be preferable that the most relevant documents appear highest in search results

The measure is normalized such that a theoretically perfect ranker could achieve a score of 1.0 on a given query (though more than one perfect ordering may be possible). To measure NDCG@10 ("NDCG *at* 10", or sometimes "NDCG *cut* 10") is to measure NDCG including only the top 10 results, which reflects real life use cases more directly. In this overview table I'm only showing measures against the TREC-DL 2021 evaluation set, calculated with the official *trec_eval* script.

Here is a description of the four checkpoints appearing in the table:

**distilbert-base** was trained on the first 5 million query-passage samples without applying any form of pruning. The model retains the original distilbert parameters: 72 total attention heads, and intermediate layers with a hidden dimension of 3072. As a sanity check on my training pipeline, I was satisfied to find that this model and a similarly trained roberta-base variant achieved scores which were within a few points of larger model results in the 2021 TREC submissions. Each of the following models were loaded from the final **distilbert-base** checkpoint and trained on an additional 5 million samples (the same 5 million for each) while pruning.

**distilbert[A]** underwent attention head pruning [(described here)](#continued-training-with-attention-head-pruning) until the final model retained 18 of its original attention heads (a 75% reduction). Prediction performance (as measured by NDCG@10) degraded by just over 10% from the starting base checkpoint, while inference time was reduced by ~38% (CPU) or ~18% (GPU).

**distilbert[B]** underwent structured magnitude pruning to reduce the hidden dimension of the models intermediate layers [(described here)](#continued-training-with-structured-magnitude-pruning) until the layers retained 307 of their original 3072 activations (a 90% reduction). This time performance only degraded 3.2% from the starting base checkpoint, while inference time was reduced by ~27% (CPU) or ~55% (GPU).

**distilbert[C]** had both head pruning and magnitude pruning methods applied during second pass training. The final model retained 22 attention heads (70% reduction) and its intermediate layers retained a hidden dimension of 512 (an 83% reduction). Prediction performance degraded just over 10% from the starting base checkpoint, and inference times were reduced by 61% (CPU) or 73% (GPU).

# Methods
Training and evaluation data came from the [2023 TREC MS MARCOv2 passage ranking task](https://microsoft.github.io/msmarco/TREC-Deep-Learning). I evaluated model checkpoints primarily with NDCG@10 against the '21 and '22 test sets using the trec_eval tool in the [Anserini toolkit](https://github.com/castorini/anserini). I ran a first pass fine tuning from the [RoBERTa](https://arxiv.org/abs/1907.11692) ([roberta-base](https://huggingface.co/roberta-base)) and [DistilBERT](https://arxiv.org/abs/1910.01108) ([distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)) checkpoints.

The first pass fine tuning followed the approach in [Passage ReRanking with BERT](https://arxiv.org/pdf/1901.04085.pdf) almost exactly, scaling the learning rate schedule down to the smaller batch sizes I can support on my hardware for the given model.

## Continued training with attention head pruning
Starting from the final checkpoint of the first pass I continued to train while incrementally pruning attention heads inspired by, but not strictly following [Voita et al.](https://arxiv.org/abs/1905.09418). Before pruning a head I measure a "confidence score" for each head, the mean of the max attention value output from the head over some number of samples, the head with the lowest score being removed. The model is exposed to further training samples before the next evaluation and pruning step. I modulated the learning rate to follow a cosine annealing schedule such that learning rate would be brought high again each time a head was pruned. This strategy has at least some support given that [learning rate rewinding](https://arxiv.org/abs/2003.02389) has outperformed standard pruning while fine tuning in other contexts. I held out the first layer's attention heads from pruning.

![Alt text](figures/pruning_lr_schedule.png)

## Continued training with structured magnitude pruning
After the attention computation, the remaining linear layers contribute the most to inference time. In particular, the intermediate projections with a higher hidden dimension which are interleaved between attention layers in the common architectures.

With some benchmarking I found that introducing unstructured sparsity and replacing the resulting layers with sparse matrices wouldn't improve runtime performance until a very high level (>97%) of sparsity was achieved.

I decided on structured pruning so the resulting projections could actually be re-sized to remove rows and columns, corresponding to "removing neurons" in the analogy. [Structured pruning in QA models](https://arxiv.org/abs/1910.06360) was reported to work well for all flavors of BERT, and the IR task could be considered fairly similar to the QA task.

I used structured pruning based on the L1 magnitude of a hidden activation's inputs (row), again with a cyclic learning rate schedule. There are other stratagies for structured pruning, including differentiable "gates" that shut off connections continuously, but this simple magnitude pruning approach has worked quite well so far.

# Visualized Runs and Next Steps
Looking at evaluation performance for the three pruning schedules highlights the point that I fixed the pruning schedules such that the models reached a certain sparsity within the modules being pruned while all being exposed to the same number of training samples. I compared "head sparsity" with the sparsity of intermediate layers.

This makes comparing the methods somewhat tenuous - for example it would appear that pruning intermediate layers is generally cheaper in terms of cost to model performance than pruning attention heads. However this is probably an artifact of the experiment design. In terms of representational "channels" available to the model, head pruning scales more steeply than intermediate pruning, because removing an attention head corresponds to more linear layers being pruned when you consider that each of the K, Q, V and output transformations are being altered. In other words, my intermediate pruning runs were "gentler" to the model, allowing the model to see the same number of samples between pruning steps while making smaller, more gradual alterations to the models structure.

In the future I would like to design another set of runs where the global sparsity rate is held fixed. I hope this will allow for a more direct comparison, and perhaps indicate which areas of the model to target first when considering performance tradeoffs.

## distilbert-base checkpoints

I'm plotting them separately for readability but do note the lower axis range for 2022 NDCG. The TREC DL-22 test queries were sampled differently, with the intention of making the queries more difficult. This is certainly reflected in the distilbert results.

![Alt text](figures/performance_vs_seen_pairs_distilbert.png)

![Alt text](figures/performance_vs_removed_flops_distilbert.png)

## roberta-base
WIP

# Reproducing these results

## Downloads

You can download the corpus and various splits of queries and qrels as suggested on the TREC DL track page. I organized the folders like:
```
~/path/to/data/msmarco-v2

       /triples.train.small.tsv
       /msmarco_v2_passage/

~/path/to/data/trec-msmarco-2023

       /2021_passage_top_100.txt
       /2021.qrels.pass.final.txt
       /2021_queries.tsv

       /2022_passage_top_100.txt
       /2022.qrels.pass.withDupes.txt
       /2022_queries.tsv
```

The corpus which should be extracted (22GB unzipped)
```shell
wget --header "X-Ms-Version: 2019-12-12" https://msmarco.blob.core.windows.net/msmarcoranking/msmarco_v2_passage.tar
```

Training triples, I only ever needed the 'small' file. (29GB unzipped):
```shell
wget --header "X-Ms-Version: 2019-12-12" https://msmarco.blob.core.windows.net/msmarcoranking/triples.train.small.tar.gz
```

The TREC DL-2021 validation set:
```shell
wget --header "X-Ms-Version: 2019-12-12" https://msmarco.z22.web.core.windows.net/msmarcoranking/2021_queries.tsv
wget --header "X-Ms-Version: 2019-12-12" https://msmarco.z22.web.core.windows.net/msmarcoranking/2021_passage_top100.txt.gz
wget --header "X-Ms-Version: 2019-12-12" https://trec.nist.gov/data/deep/2021.qrels.pass.final.txt
```

The TREC DL-2022 validation set:
```shell
wget --header "X-Ms-Version: 2019-12-12" https://msmarco.z22.web.core.windows.net/msmarcoranking/2022_queries.tsv
wget --header "X-Ms-Version: 2019-12-12" https://msmarco.z22.web.core.windows.net/msmarcoranking/2022_passage_top100.txt.gz
wget --header "X-Ms-Version: 2019-12-12" https://trec.nist.gov/data/deep/2022.qrels.pass.withDupes.txt
```


## Prefetch Validation Candidates
For simplicity of evaluation runs I prefetch the candidate passages from the collection for each of the TREC validation datasets.

```shell
python 01_prefetch_passages.py
       --corpus-dir data/msmarco-v2/msmarco_v2_passage
       --candidates-file data/trec-msmarco-2023/2021_passage_top100.txt
```

## Training
### Initial Training
You can start a new run from a base checkpoint or continue a run from a previous checkpoint, which will use optimizer and scheduler state from the checkpoint directory. Learning rate is optional and will default to the value used in Passage Re-Ranking With BERT.

```shell
python 02_train_initial.py
       --model_tag roberta-base
       --train_triples_file data/msmarco-v2/triples.train.small.tsv
       --output_dir models/roberta-base/initial-training/
       --eval_candidates_file data/trec-msmarco-2023/2021_passage_top100_hydrated.parquet
       --num_training_samples 5000000
       --num_checkpoint_samples 500000
```

### Continued training while pruning attention heads
Running head pruning in isolation on the next 5M pairs, from a checkpoint which has been trained already on the first 5M pairs:

```shell
python 03a_train_prune_att_heads.py
      --checkpoint_dir models/reranking/distilbert-base-uncased/initial-training/seen_5000000 
      --train_triples_file data/msmarco-v2/triples.train.small.tsv
      --output_dir models/reranking/distilbert-base-uncased/head-pruning-from-5000000/
      --eval_candidates_file data/trec-msmarco-2023/2021_passage_top100_hydrated.parquet
      --seen-pairs 5000000
      --target_sparsity 0.9
      --num_training_samples 5000000
```

### Continued training while pruning intermediate layers
Running intermediate structured pruning in isolation on the next 5M pairs, from a checkpoint which has been trained already on the first 5M pairs:

```shell
python 03b_train_prune_intermediates.py
      --checkpoint_dir models/reranking/distilbert-base-uncased/initial-training/seen_5000000 
      --train_triples_file data/msmarco-v2/triples.train.small.tsv
      --output_dir models/reranking/distilbert-base-uncased/intermediate-pruning-from-5000000/
      --eval_candidates_file data/trec-msmarco-2023/2021_passage_top100_hydrated.parquet
      --seen-pairs 5000000
      --target_sparsity 0.9
      --num_training_samples 5000000
```

### Continued training while using both pruning techniques
Running mixed pruning on the next 5M pairs, from a checkpoint which has been trained already on the first 5M pairs:

```shell
python 03c_train_prune_mixed.py initial
       --checkpoint_dir models/reranking/distilbert-base-uncased/initial-training/seen_5000000
       --train_triples_file data/msmarco-v2/triples.train.small.tsv
       --output_dir models/reranking/distilbert-base-uncased/mixed-pruning-from-5000000/
       --eval_candidates_file data/trec-msmarco-2023/2021_passage_top100_hydrated.parquet
       --seen-pairs 5000000
       --target_sparsity 0.9
       --num_prune_cycles 54
       --num_warmup_samples 100000
       --num_training_samples 5000000
```

The remaining scripts and notebooks are for removing pruning masks and reshaping the corresponding linear layers for models which had neuron pruning applied to them, and for generating plots.

# Full Results Table

| Model          | S  | heads | hdim | '21 NDCG@10 | '21 MAP     | '22 NDCG@10 | '22 MAP     | T(CPU) (ms)   | T(GPU) (ms)  | FLOPs (est.) |
| -----------    | -- | ----- | ---- | ----------- | ----------- | ----------- | ----------- | -----------   | -----------  | ------------ |
| distilbert-base| 5M | 72    | 3072 | 0.6018      | 0.2477      | 0.4370      | 0.0982      | 14163 +/- 129 | 1827 +/- 17  | 2.42T        |
| distilbert(A)  | 10M| 18    | 3072 | 0.5399      | 0.2311      | 0.3738      | 0.0866      | 8788 +/- 131  | 1502 +/- 10  | 1.75T        |
| distilbert(B)  | 10M| 72    | 307  | 0.5827      | 0.2427      | 0.4248      | 0.0954      | 10287 +/- 317 | 822 +/- 17   | 1.33T        |
| distilbert(C)  | 10M| 22    | 512  | 0.5382      | 0.2310      | 0.3831      | 0.0890      | 5508 +/- 347  | 499 +/- 21   | 0.74T        |
| roberta-base   | 5M | 144   | 3072 | 0.6461      | 0.2598      | 0.4672      | 0.1009      | 30187 +/- 575 | 3627 +/- 34  | 4.83T        |

