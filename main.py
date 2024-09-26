import sys, os, random, time
from xml.dom.minidom import parseString
from preprocess import read_data, normalize, tokenize, calculate_p_c, get_doc_text, update_class_docs, process_tokens
from evaluate import evaluate
from randomization import randomization_test
from multinomial_nb import train_multinomial_nb, apply_multinomial_nb
from bernoulli_nb import train_bernoulli_nb, apply_bernoulli_nb

data_path = sys.argv[1]
model_type = sys.argv[2]

start_time = time.time()

# Read the docs
documents, top_10_topics = read_data(data_path)

# Find the exact numbers of documents in train and test sets
train_doc_ids, test_doc_ids = set(), set()
all_doc_ids, all_tokens = set(), set()

for document in documents:
  doc = parseString(document)

  # Get the doc IDs
  doc_id = int(doc.documentElement.getAttribute("NEWID"))

  doc_title = doc.getElementsByTagName("TITLE")[0].firstChild.nodeValue if doc.getElementsByTagName("TITLE") else ''
  doc_body = doc.getElementsByTagName("BODY")[0].firstChild.nodeValue if doc.getElementsByTagName("BODY") else ''

  doc_final = normalize(doc_title + ' ' + doc_body)
  doc_tokens = tokenize(doc_final)

  found_in_top_10 = False
  topics_node = doc.getElementsByTagName("TOPICS") # Get the topics node
  if topics_node and (dict(doc.documentElement.attributes.items())["TOPICS"] ==  "YES"):
    split_type = doc.documentElement.getAttribute("LEWISSPLIT") # train or test

    for child_node in topics_node[0].childNodes:
      if child_node.firstChild:
        if child_node.firstChild.nodeValue in top_10_topics:
          found_in_top_10 = True
          break

    if found_in_top_10: # Add the doc id and tokens only if the topic is found in top 10
      all_tokens.update(doc_tokens)
      all_doc_ids.add(doc_id)

      if split_type == "TRAIN":
        train_doc_ids.add(doc_id)
      elif split_type == "TEST":
        test_doc_ids.add(doc_id)

# Convert train_doc_ids to a list to perform sampling and splitting
train_doc_ids = list(train_doc_ids) # len(train_doc_ids), len(test_doc_ids) # lengths are (6490, 2545)
random.seed(1) # Set the random seed
dev_doc_ids = random.sample(train_doc_ids, 1000)
train_doc_ids = [id for id in train_doc_ids if id not in dev_doc_ids]

# Convert all_tokens and all_doc_ids back to lists
all_tokens = list(all_tokens)
all_doc_ids = list(all_doc_ids)

# Initialize word_doc_dict where word_doc_dict[word][doc_id] = True if the word occurs in that doc
word_doc_dict = {token: {doc_id: False for doc_id in all_doc_ids} for token in all_tokens}
# Set the default values to False for now

train_class_docs, dev_class_docs, test_class_docs = dict(), dict(), dict()
dev_docs, test_docs = [], []
train_term_dict, train_dev_term_dict = {}, {} # train_term_dict[class][term] = term_freq
bernoulli_train_dict, bernoulli_train_dev_dict = dict(), dict() #dict[topic][term] += 1 if doc contains that term

for topic in top_10_topics:
  train_term_dict[topic] = dict()
  train_dev_term_dict[topic] = dict()
  
train_topic_dict, train_dev_topic_dict, test_topic_dict = dict(), dict(), dict()

for topic in top_10_topics:
  bernoulli_train_dict[topic], bernoulli_train_dev_dict[topic] = dict(), dict()

vocabulary, train_vocabulary, dev_vocabulary, test_vocabulary = [], [], [], []
train_count, dev_count, test_count = 0, 0, 0

for document in documents:
  doc = parseString(document)

  # Get the doc IDs
  doc_id = int(dict(doc.documentElement.attributes.items())["NEWID"])

  # Get the doc title & body
  doc_title = get_doc_text(doc, "TITLE")
  doc_body = get_doc_text(doc, "BODY")

  doc_final = normalize(f"{doc_title} {doc_body}")
  doc_tokens = tokenize(doc_final)

  if "TOPICS" in dict(doc.documentElement.attributes.items()) and (dict(doc.documentElement.attributes.items())["TOPICS"] ==  "YES"):

    topics_tag = doc.getElementsByTagName("TOPICS")[0]
    if topics_tag.firstChild:
      split_type = dict(doc.documentElement.attributes.items())["LEWISSPLIT"] # train or test
      list_of_topics = [child.firstChild.nodeValue for child in topics_tag.childNodes]
      
      for topic in list_of_topics:
        if topic in top_10_topics:
          if doc_id in train_doc_ids: # TRAIN SPLIT
            train_count += 1
            train_topic_dict[topic] = train_topic_dict.get(topic, 0) + 1
            train_dev_topic_dict[topic] = train_dev_topic_dict.get(topic, 0) + 1
      
            process_tokens(doc_tokens, vocabulary, train_vocabulary, word_doc_dict, doc_id, topic, train_term_dict, train_dev_term_dict, split="train")

            for token in list(set(doc_tokens)):
              bernoulli_train_dict[topic][token] = bernoulli_train_dict[topic].get(token, 0) + 1
              bernoulli_train_dev_dict[topic][token] = bernoulli_train_dev_dict[topic].get(token, 0) + 1 # Bernoulli train dev dict

            update_class_docs(doc_tokens, topic, train_class_docs) # Forming labeled data
            break # Topic assigned, stop further processing for this doc.

          elif doc_id in dev_doc_ids: # DEV SPLIT
            dev_count += 1
            process_tokens(doc_tokens, vocabulary, dev_vocabulary, word_doc_dict, doc_id, topic, train_term_dict, train_dev_term_dict, split="dev")
            update_class_docs(doc_tokens, topic, dev_class_docs) # Forming labeled data
            dev_docs.append((doc_id, doc_tokens, topic, " "))
            
            train_dev_topic_dict[topic] = train_dev_topic_dict.get(topic, 0) + 1
            for token in list(set(doc_tokens)):
              bernoulli_train_dev_dict[topic][token] = bernoulli_train_dev_dict[topic].get(token, 0) + 1 # Bernoulli train dev dict
      
            break # We break the loop since we have assigned a topic

          elif doc_id in test_doc_ids: # TEST SPLIT
            test_count += 1
            process_tokens(doc_tokens, vocabulary, test_vocabulary, word_doc_dict, doc_id, topic, train_term_dict, train_dev_term_dict, split="test")
            test_topic_dict[topic] = test_topic_dict.get(topic, 0) + 1
            update_class_docs(doc_tokens, topic, test_class_docs) # Forming labeled data
            test_docs.append((doc_id, doc_tokens, topic, " "))
            break # We break the loop since we have assigned a topic

dev_test_vocabulary = dev_vocabulary + test_vocabulary
p_c = calculate_p_c(train_topic_dict)

# Precompute the unique words from the vocabulary
unique_vocabulary = set(vocabulary)
for topic in train_term_dict.keys(): # Iterate over topics
    for word in unique_vocabulary:
        # Use setdefault to ensure the word exists in the dictionaries with a default value of 0
        train_term_dict[topic].setdefault(word, 0)
        train_dev_term_dict[topic].setdefault(word, 0)

if (model_type == "multinomial-nb") or (model_type == "all"):

  if model_type == "multinomial-nb":

    alphas = [0.5, 1, 2]
    for alpha in alphas:

      class_conditional_dict = train_multinomial_nb(vocabulary, train_term_dict, train_class_docs, dev_test_vocabulary, alpha)
      targets_multinomial, preds_multinomial = apply_multinomial_nb(dev_docs, dev_class_docs, p_c, class_conditional_dict)
      print(f"\nMultinomial Naive Bayes with alpha = {alpha}. Performance on the dev set:")
      evaluate(targets_multinomial, preds_multinomial)

  # Train with train + dev set using the best hyperparameter (alpha) and evaluate the performance on the test set

  p_c_train_dev = calculate_p_c(train_dev_topic_dict)
  class_conditional_dict = train_multinomial_nb(vocabulary, train_dev_term_dict, train_class_docs, dev_test_vocabulary, alpha=0.5)
  targets_multinomial_best, preds_multinomial_best = apply_multinomial_nb(test_docs, test_class_docs, p_c_train_dev, class_conditional_dict)

  print("\nMultinomial Naive Bayes with alpha = 0.5. Performance on the test set:")
  evaluate(targets_multinomial_best, preds_multinomial_best)

# BERNOULLI

if (model_type == "bernoulli-nb") or (model_type == "all"):

  for word in vocabulary:
    for topic in top_10_topics:
      bernoulli_train_dict[topic].setdefault(word, 0)
      bernoulli_train_dev_dict[topic].setdefault(word, 0)
  
  if model_type == "bernoulli-nb":

    alphas = [0.5, 1, 2]
    for alpha in alphas:
      bernoulli_class_conditional_dict = train_bernoulli_nb(vocabulary, train_class_docs, train_topic_dict, bernoulli_train_dict, alpha)
      targets_bernoulli, preds_bernoulli = apply_bernoulli_nb(dev_docs, top_10_topics, unique_vocabulary, word_doc_dict, p_c, bernoulli_class_conditional_dict)
      print(f"\nMultivariate Bernoulli Naive Bayes with alpha = {alpha}. Performance on the dev set:")
      evaluate(targets_bernoulli, preds_bernoulli)  

  # Train with train + dev set using the best hyperparameter (alpha) and evaluate the performance on the test set

  p_c_train_dev = calculate_p_c(train_dev_topic_dict)
  bernoulli_class_conditional_dict  = train_bernoulli_nb(vocabulary, train_class_docs, train_dev_topic_dict, bernoulli_train_dev_dict, alpha=0.5)
  
  print("\nMultivariate Bernoulli Naive Bayes with alpha = 0.5. Performance on the test set:")
  targets_bernoulli_best, preds_bernoulli_best = apply_bernoulli_nb(test_docs, top_10_topics, unique_vocabulary, word_doc_dict, p_c_train_dev, bernoulli_class_conditional_dict)
  evaluate(targets_bernoulli_best, preds_bernoulli_best)

if model_type == "all":
  print("\nRandomization test:")
  randomization_test(targets_multinomial_best, preds_multinomial_best, preds_bernoulli_best)

end_time = time.time()
print(f"\nElapsed time: {end_time - start_time} seconds.")