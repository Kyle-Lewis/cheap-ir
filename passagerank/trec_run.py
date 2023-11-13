import csv

import torch


def trec_eval_run(
    model,
    trec_val_dataloader,
    output_dir,
    affix,
    pbar=None
):
    
    model.eval()

    rows = []
    qi = 0
    num_eval_queries = len(trec_val_dataloader)
    for batch in trec_val_dataloader:

        qid = batch.pop('qid')[0]
        input_pids = batch.pop('pid')
        

        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        
        qi += 1
        if pbar is not None:
            pbar.set_postfix({"EVAL": "{}/{}".format(qi, num_eval_queries)})
        else:
            print("EVAL ({: 3}/{})".format(qi, num_eval_queries), end="\r")
        
        scores = outputs.logits[:,1].cpu()
        model_ranked_order = torch.argsort(scores, descending=True)

        for i in range(len(scores)):

            ordered_idx = model_ranked_order[i].item()

            rank = i + 1
            pid = input_pids[ordered_idx]
            score = scores[ordered_idx].item()

            rows.append([qid, "Q0", pid, rank, score, "testrun"])
    
    print()
    with open(str(output_dir) + '/trec_{}_run.tsv'.format(affix), 'w+') as outfile:
        writer = csv.writer(outfile, delimiter='\t')
        for row in rows:
            writer.writerow(row)
    
    model.train()