import random

def precision(TP, FP):
  if (TP + FP) == 0: return 0
  else: return TP / (TP + FP)

def recall(TP, FN):
  if (TP + FN) == 0: return 0
  else: return TP / (TP + FN)

def f_score(prec, recall):
  if (prec + recall) == 0: return 0
  else: return ((2 * prec * recall) / (prec + recall))

def macro_f(f_dict):
  return sum(f_dict.values()) / len(f_dict.values())

def micro_f(TP, FP, FN):
  prec = precision(TP, FP)
  rec = recall(TP, FN)
  return f_score(prec, rec)

def evaluate(targets, preds, verbose=True):
  if (len(list(set(targets))) != 10) and (len(list(set(preds))) != 10): print("WARNING")
  tp, fp, fn, tn = dict(), dict(), dict(), dict()
  for topic in list(set(targets)):
    tp[topic], fp[topic], fn[topic], tn[topic] = 0, 0, 0, 0

  for target, pred in zip(targets, preds):
    if target == pred:
      tp[target] += 1
      for topic in list(set(targets)):
        if topic != target:
          tn[topic] += 1

    elif target != pred:
      fp[pred] += 1
      fn[target] += 1
      for topic in list(set(targets)):
        if (pred != topic) and (target != topic):
          tn[topic] += 1
  
  prec_dict, recall_dict, f_dict = dict(), dict(), dict()
  for topic in list(set(targets)):
    prec_dict[topic] = precision(tp[topic], fp[topic])
    recall_dict[topic] = recall(tp[topic], fn[topic])
    f_dict[topic] = f_score(prec_dict[topic], recall_dict[topic])

  TP, FP, FN, TN = sum(tp.values()), sum(fp.values()), sum(fn.values()), sum(tn.values())
  
  if verbose:
    print(f"Macro F-score: {macro_f(f_dict):.3f}")
    print(f"Micro F-score: {micro_f(TP, FP, FN):.3f}")

  return macro_f(f_dict)