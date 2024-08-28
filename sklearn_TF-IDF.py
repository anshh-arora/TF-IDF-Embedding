# import required module
from sklearn.feature_extraction.text import TfidfVectorizer

#assign documnents
d0 = "The cat sat on the mat"
d1 = "The dog sat on the wall"
d2 = "They are playing"

#merge documnet into a single corpus
string  = [d0,d1,d2]
print(f"string :{string}")

# create a object
tfidf = TfidfVectorizer()

#get tf-df values
result = tfidf.fit_transform(string)
print(f"result :\n{result}")

"""# get idf values
print("\n idf values: ")
for ele1, ele2 in zip(tfidf.get_feature_names(), tfidf.idf_):
    print(ele1, ':', ele2)
"""

#get indexing
print("\n Word Indexes: ")
print(tfidf.vocabulary_)

#display tf-idf values
print("\n tf-idf value :")
print(result)

# in matrix form
print("\n tf-idf values in matrix form :")
print(result.toarray())