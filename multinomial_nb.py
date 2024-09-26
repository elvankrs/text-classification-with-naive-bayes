import math

def train_multinomial_nb(vocabulary, term_dict, train_class_docs, dev_test_vocabulary, alpha):
  # Calculate P(w_k | c_j) terms
  """
  Text_j = single doc containing all docs_j
  for each word w_k in Vocabulary:
    n_k = # of occurrences of w_k in Text_j
    n = number of words (tokens) in Text_j

  """
  class_conditional_dict = dict()
  V = len(list(set(vocabulary)))

  for topic in train_class_docs.keys():
    class_conditional_dict[topic] = dict()
    n = sum(term_dict[topic].values())
    for w_k in list(set(dev_test_vocabulary)):
      n_k = term_dict[topic][w_k]
      class_conditional_dict[topic][w_k] = (n_k + alpha) / (n + alpha * V)

  return class_conditional_dict


def apply_multinomial_nb(docs, test_class_docs, p_c, p_w_given_c):
  ctr=0
  targets, preds = [], []
  for doc in docs:
    doc_tokens = doc[1] # Get the doc tokens
    score = dict()
    for topic in test_class_docs.keys():
      score[topic] = math.log(p_c[topic])
      for word in doc_tokens:
        score[topic] += math.log(p_w_given_c[topic][word])

    predicted_topic = max(score, key=score.get)
    preds.append(predicted_topic)
    targets.append(doc[2])
  
  return targets, preds