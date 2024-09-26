import math

def train_bernoulli_nb(vocabulary, train_class_docs, topic_dict, bernoulli_dict, alpha):

    # Calculate P(w_k | c_j) terms
    """
    Text_j = single doc containing all docs_j
    for each word w_k in Vocabulary:
      n_k = # of occurrences of w_k in Text_j
      n = number of words (tokens) in Text_j

    """
    bernoulli_class_conditional_dict = dict()
    V = len(list(set(vocabulary)))
    N = sum(topic_dict.values()) 

    for topic in train_class_docs.keys():
      bernoulli_class_conditional_dict[topic] = dict()
      N_c = sum(topic_dict.values())

      for t in list(set(vocabulary)):
        N_ct = bernoulli_dict[topic][t]        
        bernoulli_class_conditional_dict[topic][t] = (N_ct + alpha) / (N_c + 2 * alpha) 
           
    return bernoulli_class_conditional_dict


def apply_bernoulli_nb(docs, top_10_topics, unique_vocabulary, word_doc_dict, p_c, p_w_given_c):
  targets, preds = [], []
  for doc in docs:
    doc_id, doc_tokens = doc[0], doc[1] # Get the doc tokens and doc id

    score = dict()
    for topic in top_10_topics:
      score[topic] = math.log(p_c[topic])
      for word in unique_vocabulary:
        if word_doc_dict[word][doc_id] == True:
          score[topic] += math.log(p_w_given_c[topic][word])
        else:
          score[topic] += math.log(1 - p_w_given_c[topic][word])

    predicted_topic = max(score, key=score.get)
    preds.append(predicted_topic)
    targets.append(doc[2])

  return targets, preds