import numpy as np
from sklearn.metrics import top_k_accuracy_score
import torch
# top_k_accuracy_score(y_true, y_score, k=2)


def top_k(
        predictions,
        attr_truth,
        obj_truth,
        pair_truth,
        train_pairs,
        raw=False,
        topk=1):
    # Go to CPU

    attr_truth, obj_truth, pair_truth = (
        attr_truth.to("cpu"),
        obj_truth.to("cpu"),
        pair_truth.to("cpu"),
    )

    pairs = list(zip(list(attr_truth.numpy()), list(obj_truth.numpy())))

    attr_match = (
            attr_truth.unsqueeze(1).repeat(1, topk) == predictions[0][:, :topk]
    )
    obj_match = (
            obj_truth.unsqueeze(1).repeat(1, topk) == predictions[1][:, :topk]
    )

    seen_ind, unseen_ind = [], []
    for i in range(len(attr_truth)):
        if pairs[i] in train_pairs:
            seen_ind.append(i)
        else:
            unseen_ind.append(i)

    seen_ind, unseen_ind = torch.LongTensor(seen_ind), torch.LongTensor(
        unseen_ind)

    seen_attr_match, unseen_attr_match = attr_match[seen_ind], attr_match[unseen_ind]
    seen_obj_match, unseen_obj_match = obj_match[seen_ind], obj_match[unseen_ind]

    if raw:
        return seen_attr_match, seen_obj_match, unseen_attr_match, unseen_obj_match
    else:
        return seen_attr_match/len(seen_ind), seen_obj_match/len(seen_ind),\
                unseen_attr_match/len(unseen_ind), unseen_obj_match/len(unseen_ind)