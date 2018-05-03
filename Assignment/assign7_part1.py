from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from scipy.stats import entropy
from matplotlib import pyplot as plt

word_types = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
topic_word = {}
topic_word[0] = word_types[0:7] #7
topic_word[1] = word_types[7:14] #7
topic_word[2] = word_types[14:20] #6


def createDocs(topic_word, num_docs, words_per_doc):
    docs = list()
    word_count = [[0 for i in range(20)] for j in range(3)]

    mean_topic_entropy = 0.0
    mean_word_entropy = 0.0

    for d in range(200):
        word_list = ""
        word_sampling_per_topic = list()
        word_sampling_per_topic.append(np.random.dirichlet([0.01 for l in range(len(topic_word[0]))]))
        word_sampling_per_topic.append(np.random.dirichlet([0.01 for l in range(len(topic_word[1]))]))
        word_sampling_per_topic.append(np.random.dirichlet([0.01 for l in range(len(topic_word[2]))]))

        topic_sampling = np.random.dirichlet([0.1 for a in range(3)])

        words_per_topic = np.random.multinomial(words_per_doc, topic_sampling, size=1)[0] # number of words per topic to be selected for a doc

        mean_topic_entropy = mean_topic_entropy + entropy(topic_sampling)
        mean_word_entropy = mean_word_entropy + entropy(word_sampling_per_topic[0]) \
                            + entropy(word_sampling_per_topic[1]) + entropy(word_sampling_per_topic[2])

        for j in range(len(topic_sampling)):
            for i in range(words_per_topic[j]):
                word_index = np.random.multinomial(words_per_topic[j], word_sampling_per_topic[j]).argmax()
                word = topic_word[j][word_index]
                word_list = word_list + word + " "
                word_count[j][j*7+word_index] = word_count[j][word_index] + 1
        docs.append(word_list)
    we = mean_word_entropy/3.0
    return docs, word_count, we, mean_topic_entropy/200.0


def print_top_words(model, feature_names, n_top_words):
    word_dist_topics = list()
    topics = [0, 0, 0]
    topic_words = {}
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])

        word_dist = list()
        topic_dist = list()

        # word probability
        for i in range(len(topic)):
            word_dist.append(topic[i]/sum(topic))
            if i in range(7):
                topics[0] = topics[0] + 1.0/200.0
            if i in range(7,14):
                topics[1] = topics[1] + 1.0/200.0
            if i in range(14,20):
                topics[2] = topics[2] + 1.0/200.0

        word_dist_topics.append(word_dist)
        word_idx = np.argsort(topic)[::-1][:n_top_words]
        topic_words[topic_idx] = [feature_names[i] for i in word_idx]

        print(message)
    return word_dist_topics, topics


# def entropy(sampling):
#     ent = 0.0
#     for i in sampling:
#         if i is not 0:
#             ent = ent + i * np.log(i)
#     return -1 * ent


num_topics = 3
num_docs = 200
words_per_doc = 50
alpha = 0.1
beta = 0.01

# created a model
lda = LatentDirichletAllocation(n_components=num_topics, doc_topic_prior=alpha, topic_word_prior=beta, learning_method='online')

#create docs
docs, word_count, word_ent, topic_ent = createDocs(topic_word, num_docs, words_per_doc)
# print word_ent
print topic_ent

# recovered topics
count_vectorizer = CountVectorizer(max_features=20, analyzer='char', stop_words=None, max_df=0.999)
cv = count_vectorizer.fit_transform(docs)
k = cv.todense()
# lda.fit(cv)

# change alpha
y = list()
x = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
for i in range(10):
    lda = LatentDirichletAllocation(n_components=num_topics, doc_topic_prior=alpha, topic_word_prior=float((i+1)*5), learning_method='online')
    lda.fit(cv)
    word_dist, topics = print_top_words(lda, word_types, 20)
    word_sampling = 0
    for j in range(3):
        arr = word_dist[j]
        word_sampling = word_sampling + entropy(arr)
    print word_sampling/3.0
    y.append(word_sampling/3.0)
    # y.append(entropy(topics))

plt.plot(x, y)
plt.xlabel("Alpha")
plt.ylabel("Entropy")
plt.title("Entropy of Topic Distribution")
plt.show()
# true topic distribution
for i in range(3):
    array = word_count[i]
    arr = np.array(array).argsort()[::-1]
    print "Topic " + str(i)
    lst = list()
    for c in arr:
        lst.append(word_types[c])
    print lst

# word_dist, topics = print_top_words(lda, word_types, 20)
# word_sampling = 0
# for i in range(3):
#     arr = word_dist[i]
#     word_sampling = word_sampling + entropy(arr)
# print word_sampling/200.0
