import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


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
#print(samples)

tagged_data = [TaggedDocument(words=d.split(), tags=[i]) for i, d in enumerate(samples)]
print("Tagged Data: ",tagged_data)

model = Doc2Vec(vector_size=20, min_count=0, epochs=10)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs) #total_examples是
print("corpus_count: ", model.corpus_count)
print("corpus_total_words: ", model.corpus_total_words)

test_data = samples[3].split()
print("Test Data: ", test_data)
v1 = model.infer_vector(test_data)
sims = model.dv.most_similar([v1], topn=len(model.dv)) #topn是返回相似文档的数量
print(sims)
print(len(model.dv))