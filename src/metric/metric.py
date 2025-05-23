import random
import os
import numpy as np
import json
import spacy
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import sklearn


nlp = spacy.load("en_core_web_sm")

def few_metric_builder(args, tokenizer, split):
    # fname = os.path.join(data_dir, '{}.json'.format(split))
    if split == 'dev' and args.is_few:
        split = split + '_few'
    if split == 'train' and args.is_few:
        split = split + '_few' + args.few
    input_file = os.path.join(args.data_dir, f"{args.data_name}_{split}.json")
    with open(input_file, "r") as f:
        datas = json.load(f)
    # concept_sets = []
    # for data in datas.values():
        # data = eval(data)
        # concept_sets.append(data["concepts"].split(', '))

    def get_metrics(yd_true, yd_pred):
        metrics = {}
        concepts = set(yd_true.keys()).intersection(set(yd_pred.keys()))
        y_true = [yd_true[c][0] for c in concepts]
        y_pred = [yd_pred[c] for c in concepts]
        # print(y_true)
        t = sklearn.metrics.classification_report(y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False)
        print(t)
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='micro')
        metrics['recall'] = recall_score(y_true, y_pred, average='micro')
        metrics['micro_f1_score'] = f1_score(y_true, y_pred, average='micro')
        metrics['macro_f1_score'] = f1_score(y_true, y_pred, average='macro')
        conf_mat = confusion_matrix(y_true, y_pred, labels=['positive', 'neutral', 'negative'])
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=[0, 1, 2])
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        save_path = os.path.join(args.output_dir, "conf_matrix{}.png".format(datetime.now().strftime("%m-%d %H:%M")))
        # print(save_path)
        plt.savefig(save_path)
        plt.clf()
        # plt.show()
        # metrics = classification_report(y_true, y_pred, output_dict=True)
        return metrics

    def evaluator(gts, res):
        eval = {}
        # =================================================
        # Set up scorers, 
        # =================================================
        # print('tokenization...')
        # Todo: use Spacy for tokenization
        # gts = tokenize(gts)
        # res = tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        # print('setting up scorers...')

        # =================================================
        # Compute scores
        # =================================================
        # for scorer, method in scorers:
        #     print('computing %s score...' % (scorer.method()))
        #     score, scores = scorer.compute_score(gts, res)
        #     if type(method) == list:
        #         for sc, scs, m in zip(score, scores, method):
        #             eval[m] = sc
        #             # print("%s: %0.3f" % (m, sc))
        #     else:
        #         eval[method] = score
        #         # print("%s: %0.3f" % (method, score))

        # return eval

    def compute_metrics(pred):
        # print('pred'*6)
        """Utility to compute ROUGE during training."""
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
        # print(pred_ids[0])
        # input()

        # All special tokens are removed.
        pred_ids[pred_ids == -100] = tokenizer.pad_token_id
        # for i, pred_id in enumerate(pred_ids):
        #     try:
        #         tokenizer.batch_decode([pred_id], skip_special_tokens=True)
        #     except:
        #         print("pred_id", i, pred_id)

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        # print(pred_str)
        # input()
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        # for i, labels_id in enumerate(labels_ids):
        #     try:
        #         tokenizer.batch_decode([labels_id], skip_special_tokens=True)
        #     except:
        #        print("labels_id", i, labels_id)

        # print("========")
        # print(labels_ids.shape)
        # print(labels_ids)

        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        idxs = random.sample(range(len(pred_str)), 10)
        for idx in idxs:
            print("========== {} ==========".format(idx))
            print("label: {}".format(label_str[idx]))
            print("pred: {}".format(pred_str[idx]))
        print('label positive: {}'.format(label_str.count('positive')))
        print('label neutral: {}'.format(label_str.count('neutral')))
        print('label negative: {}'.format(label_str.count('negative')))
        print('pred positive: {}'.format(pred_str.count('positive')))
        print('pred neutral: {}'.format(pred_str.count('neutral')))
        print('pred negative: {}'.format(pred_str.count('negative')))
        # Compute the metric.
        gts = {}
        res = {}
        i = -1
        # for key_line, gts_line in zip(key_lines, gts_lines):
        for data in datas.values():
            key = '#'.join(data["concepts"].rstrip().split(', '))
            if key not in gts:
                i += 1
                if i >= len(pred_str):
                    break
                gts[key] = []
                #TODO: change target
                # target = data["concepts"].rstrip('\n')
                gts[key].append(data["label"].rstrip('\n'))
                res[key] = pred_str[i].rstrip('\n')
                # res[key].append(pred_str[i].rstrip('\n'))
            else:
                gts[key].append(data["label"].rstrip('\n'))
        res = get_metrics(gts, res)
        avg = []
        for k, v in res.items():
            avg.append(v*100)
        res["average"] = np.mean(avg)
        # print(res)
        return res
    return compute_metrics
