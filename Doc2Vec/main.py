import numpy as np
from collections import Counter
from scipy.special import expit

#Samples
samples = """
Brazil police arrest third suspect in killings of Dom Phillips and Bruno Pereira|
‘Ukraine will definitely win’ says president on visit to Mykolaiv|
Rail strikes will 'punish millions of innocent people', govt warns - but Labour claims PM wants it to happen|
Unions 'bribing' workers to strike|
Public told to remain indoors as UK hit by ‘huge cluster’ of thunderstorms|
‘My foot got caught’: Biden falls off bike in Rehoboth Beach|
Former boxer told teen barmaid 'call me when you turn 18' before causing pub chaos|
Decisive majority of Tory voters back leaving ECHR over Rwanda policy|
Boris Johnson defends plans to electronically tag asylum seekers|
Tory rebels warn by-election ‘disaster’ will pile pressure on Boris Johnson|
""".split("|")

class Vocabulary:
    def __init__(self, samples, min_count):
        sentences = [ samples[i].split() for i in range(len(samples))]
        words = [word for s in sentences for word in s]
        self.sentences = sentences
        self.freqs = {w:n for w, n in Counter(words).items() if n >= min_count}
        self.words = sorted(self.freqs.keys())
        self.word2idx = {w:i for i, w in enumerate(self.words)}
        self.probs = np.power(np.array([self.freqs[w] for w in self.words]), 0.75)
        self.probs /= np.sum(self.probs)
        threshold_count = 0.001 * len(self.words)
        self.sample_threshold = {w : (np.sqrt(self.freqs[w] / threshold_count) + 1) * (threshold_count / self.freqs[w]) for w in self.words}

    def sample(self, num_negative_samples, positive_index):
        sample_ids = np.random.choice(a=self.probs.shape[0], size=num_negative_samples + 1, p=self.probs, replace=False).tolist()
        sample_ids = [sample_id for sample_id in sample_ids if sample_id != positive_index]
        if len(sample_ids) > num_negative_samples:
            sample_ids.remove(sample_ids[num_negative_samples])
        sample_ids.insert(0, positive_index)
        return sample_ids

def train_cbow_pair(word, l1, alpha, learn_hidden=True):
    neu1e = np.zeros(l1.shape)
    word_index = vocab.word2idx[word]
    # 按照单词出现的概率 负采样的单词也是随机的
    word_indices = vocab.sample(num_negative_samples, word_index)
    l2b = hidden_vectors[word_indices]
    prod_term = np.dot(l1, l2b.T)
    fb = expit(prod_term) #the predicted probabilities of the negatively sampled words
    gb = (negative_labels - fb) * alpha #the difference between the true labels for negative examples (negative_labels) and the predicted probabilities
    if learn_hidden:
        hidden_vectors[word_indices] += np.outer(gb, l1)
    neu1e += np.dot(gb, l2b)
    return neu1e
    
def train_document_dm(doc_words, doctag_indexes, learn_doctags=True, learn_words=True, learn_hidden=True):
    window_size = 5
    # 按照单词的频率 随机mask掉句子中部分单词
    doc_words = [w for w in doc_words if vocab.sample_threshold[w] > np.random.random()]
    for pos, word in enumerate(doc_words):
        reduced_window = np.random.randint(0, window_size - 1)
        # 随机窗口的大小
        win_size = window_size - reduced_window
        start = max(0, win_size)
        window_pos = enumerate(doc_words[start:(pos + win_size + 1)], start)
        word_indexes = [vocab.word2idx[word2] for pos2, word2 in window_pos if pos2 != pos]
        l1 = np.sum(word_vectors[word_indexes], axis=0) + np.sum(doctag_vectors[doctag_indexes], axis=0)
        count = len(word_indexes) + len(doctag_indexes)
        l1 /= count
        neu1e = train_cbow_pair(word, l1, 0.01, learn_hidden)

        if learn_doctags:
            for i in doctag_indexes:
                doctag_vectors[i] += neu1e  # * doctag_locks[i]
        if learn_words:
            for i in word_indexes:
                word_vectors[i] += neu1e  # * word_locks[i]


#Train
vocab = Vocabulary(samples, 1)
vector_size = 20
num_negative_samples = 5

word_vectors = np.random.random((len(vocab.words), vector_size))
doctag_vectors = np.random.random((len(vocab.sentences), vector_size))
hidden_vectors = np.random.random((len(vocab.words), vector_size))

negative_labels = [0 for i in range(num_negative_samples)]
negative_labels.insert(0, 1)
negative_labels = np.array(negative_labels)

epoches = 10
for epoch in range(epoches):
    for pos, words in enumerate(vocab.sentences):
        train_document_dm(words, [pos])

saved_doctag = doctag_vectors

#Test
index = 3
for e in range(epoches):
    train_document_dm(vocab.sentences[index], [0], learn_words= False, learn_hidden= False)

doctag_vectors = np.random.random((1,vector_size))
vec =doctag_vectors[0]
for i in range(len(vocab.sentences)):
    doc = saved_doctag[i]
    # Calculate the cosine similarity between doc and vec
    dot_product = np.dot(doc, vec)
    doc_norm = np.linalg.norm(doc)
    vec_norm = np.linalg.norm(vec)
    # Calculate the cosine similarity
    cosine_similarity = dot_product / (doc_norm * vec_norm)
    print(f"Sentence {i}: {cosine_similarity}")

