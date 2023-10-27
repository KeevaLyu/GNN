from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk

#nltk.download()

#Train
data = ["I love machine learning. Its awesome.",
        "I love coding in python",
        "I love building chatbots",
        "they chat amagingly well"]

tagged_data = [TaggedDocument(words=word_tokenize(d.lower()), tags=[str(i)]) for i, d in enumerate(data)]
print("Tagged data: ", tagged_data)

epochs = 10
vec_size = 10
alpha = 0.025
model = Doc2Vec(vector_size=vec_size, alpha=alpha, min_count=1, dm=1)
model.build_vocab(tagged_data)
for epoch in range(epochs):
    model.train(tagged_data, total_words=model.corpus_total_words, epochs=model.epochs)
    model.alpha -= 0.0002
    model.min_alpha = model.alpha
#model.save("d2v.model")
print("Finishing")


#Test
#model = Doc2Vec.load("d2v.model")
test_datas = ["I love chatbots",
              "I love chatbots"]
tests = [word_tokenize(test_data.lower()) for test_data in test_datas]
print(tests)
vecs = [model.infer_vector(test) for test in tests]
print("vec_infer: ", vecs)
similar_doc = model.dv.most_similar('1')
print(similar_doc)
print(model.dv['1'])