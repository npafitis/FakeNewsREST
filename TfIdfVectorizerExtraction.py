from sklearn.feature_extraction.text import TfidfTransformer

from CountVectorizerExtraction import CountVectorizerExtractor


class TfIdfVectorizerExtractor:
    tfidf_vectorizer = TfidfTransformer(use_idf=True, smooth_idf=True)
    tfidf_vectorizer_normalized = TfidfTransformer(use_idf=True, smooth_idf=True)
    tfidf_vectorizer_ngram = TfidfTransformer(use_idf=True, smooth_idf=True)
    tfidf_vectorizer_ngram_normalized = TfidfTransformer(use_idf=True, smooth_idf=True)
    tfidf_vectorizer_pos_tag = TfidfTransformer(use_idf=True, smooth_idf=True)
    tfidf_matrix = {}
    tfidf_matrix_normalized = {}
    tfidf_matrix_ngram = {}
    tfidf_matrix_ngram_normalized = {}
    tfidf_matrix_pos_tag = {}

    def __init__(self, CountVectorizer):
        print(CountVectorizer.count_vectorizer.vocabulary_)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(CountVectorizer.td_matrix)
        self.tfidf_matrix_normalized = self.tfidf_vectorizer_normalized.fit_transform(CountVectorizer.td_matrix_normalized)
        self.tfidf_matrix_ngram = self.tfidf_vectorizer_ngram.fit_transform(CountVectorizer.td_matrix_ngram)
        self.tfidf_matrix_ngram_normalized = self.tfidf_vectorizer_ngram_normalized.fit_transform(
            CountVectorizer.td_matrix_ngram_normalized)
        self.tfidf_matrix_pos_tag = self.tfidf_vectorizer_pos_tag.fit_transform(CountVectorizer.td_matrix_pos_tag)

    def test_tfidf_matrix(self):
        print(self.tfidf_matrix.shape)
        print(self.tfidf_matrix.A[:10])
        print(self.tfidf_matrix)


if __name__ == '__main__':
    cve = CountVectorizerExtractor()
    tve = TfIdfVectorizerExtractor(cve)
    tve.test_tfidf_matrix()
