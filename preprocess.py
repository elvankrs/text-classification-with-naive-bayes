import os, re, string
from xml.dom.minidom import parseString
from collections import Counter

def find_common_topics(documents):

  topic_counter = Counter()

  for document in documents:
      doc = parseString(document)

      topics_node = doc.getElementsByTagName("TOPICS")
      
      # Check if the TOPICS element exists and is marked as "YES"
      if topics_node and dict(doc.documentElement.attributes.items())["TOPICS"] == "YES":
          for child_node in topics_node[0].childNodes:
              if child_node.firstChild:
                  topic = child_node.firstChild.nodeValue
                  topic_counter[topic] += 1

  # Get the 10 most common topics
  top_10_topics = [topic for topic, count in topic_counter.most_common(10)]

  return top_10_topics

def read_data(data_path):

  rm_pattern = re.compile(r"&#\d*;")
  rm2_pattern = re.compile(r"\n Reuter\n")
  doc_pattern = re.compile(r"<REUTERS.*?<\/REUTERS>", re.S)

  documents = []
  for file in os.listdir(data_path):
      
    if file.endswith(".sgm"):
        
      # Read each sgm file
      file_name = os.path.join(data_path, file)
      f = open(file_name, 'r', encoding='latin-1', errors='ignore')
      data_file = f.read()
      f.close()

      data_file = rm_pattern.sub('', data_file) # Remove &#0 since XML does not support these characters 
      data_file = rm2_pattern.sub('', data_file) # Remove "Reuter" which occurs at the end of each BODY field

      file_documents = doc_pattern.findall(data_file) # Extract the documents in the file
      documents += file_documents # Add file documents to all documents

  top_10_topics = find_common_topics(documents)

  return documents, top_10_topics

def get_doc_text(doc, tag_name):
  # Retrieve text content from an XML tag
  if len(doc.getElementsByTagName(tag_name)) != 0 and doc.getElementsByTagName(tag_name)[0].firstChild:
      return doc.getElementsByTagName(tag_name)[0].firstChild.nodeValue
  return ''

def update_class_docs(doc_tokens, topic, split_class_docs):
  # Update the split_class_docs dictionary -> it can be train_class_docs, dev_class_docs, test_class_docs
  if topic in split_class_docs.keys():
    split_class_docs[topic].extend(doc_tokens) # Forming labeled data
  else:
    split_class_docs[topic] = doc_tokens
  
def process_tokens(doc_tokens, vocabulary, split_vocabulary, word_doc_dict, doc_id, topic, train_term_dict, train_dev_term_dict, split):
  # Process the tokens by updating vocabulary and word doc dict
  # split_vocabulary -> can be train_vocabulary, dev_vocabulary, test_vocabulary
  for token in doc_tokens: # Adding all the terms to the big vocabulary
    vocabulary.append(token)
    split_vocabulary.append(token)
    word_doc_dict[token][doc_id] = True

    if split=="train":
      train_term_dict[topic][token] = train_term_dict[topic].get(token, 0) + 1
      train_dev_term_dict[topic][token] = train_dev_term_dict[topic].get(token, 0) + 1 # train dev term dict
    elif split=="dev":
      train_term_dict[topic].setdefault(token, 0) # train term dict
      train_dev_term_dict[topic][token] = train_dev_term_dict[topic].get(token, 0) + 1 # train dev term dict
    elif split=="test":
      train_term_dict[topic].setdefault(token, 0) # train term dict
      train_dev_term_dict[topic].setdefault(token, 0) # train dev term dict


def normalize(text):
  text = text.translate(str.maketrans('', '', string.punctuation)) # Punctuation removal
  text = re.sub(r'\d+','', text) # Removing numbers
  text = text.lower()  # Case folding
  f = open("stopwords.txt", "r")
  stopwords = f.read().splitlines()
  f.close()
  tokens = [word for word in text.split() if word.lower() not in stopwords] # Removing stopwords
  text = " ".join(tokens)
  return text

def tokenize(text):
  text = text.replace('\n', ' ')
  tokens = text.split()
  return tokens

def calculate_p_c(dict):
  # Calculate P(c_j) terms
  p_c = {} # Prior class probabilities
  for topic, freq in dict.items():
    p_c[topic] = freq / sum(dict.values())

  return p_c