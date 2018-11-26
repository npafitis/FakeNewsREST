# string to test
import numpy
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import CsvReader
from CountVectorizerExtraction import CountVectorizerExtractor
from TfIdfVectorizerExtraction import TfIdfVectorizerExtractor


class NaiveBayesClassifier:

    def build_naive_bayes_classifier_cv(self, cve):
        self.nb_pipeline_cv = Pipeline([
            ('NBCV', cve.count_vectorizer),
            ('nb_clf', MultinomialNB())])

        self.nb_pipeline_cv.fit(cve.df['articleBody'], cve.df['Stance'])
        self.predicted_nb_cv = self.nb_pipeline_cv.predict(
            CsvReader.test_df['articleBody'])
        self.nb_mean_cv = numpy.mean(self.predicted_nb_cv == cve.df['Stance'])
        print('Naive Bayes with Count Vector', self.nb_mean_cv)

    def build_naive_bayes_classifier_tfidf(self, tfidf, cve):
        self.nb_pipeline_tfidf = Pipeline([
            ('NBCV', tfidf.tfidf_vectorizer),
            ('nb_clf', MultinomialNB())])

        self.nb_pipeline_tfidf.fit(cve.df['articleBody'],
                                   cve.df['Stance'])
        self.predicted_nb_tfidf = self.nb_pipeline_tfidf.predict(
            CsvReader.test_df['articleBody'])
        self.nb_mean_tfidf = numpy.mean(self.predicted_nb_tfidf == cve.df['Stance'])
        print('Naive Bayes with TFIDF', self.nb_mean_tfidf)

    def build_naive_bayes_classifier_cv_normalized(self, cve):
        self.nb_pipeline_cv_normalized = Pipeline([
            ('NBCV', cve.count_vectorizer_normalized),
            ('nb_clf', MultinomialNB())])

        self.nb_pipeline_cv_normalized.fit(cve.df_normalized['articleBody'],
                                           cve.df['Stance'])
        self.predicted_nb_cv_normalized = self.nb_pipeline_cv_normalized.predict(
            CsvReader.test_df_normalized['articleBody'])
        self.nb_mean_cv_normalized = numpy.mean(self.predicted_nb_cv_normalized == CsvReader.test_df['Stance'])
        print('Naive Bayes with Count Vector and normalized', self.nb_mean_cv_normalized)

    def build_naive_bayes_classifier_tfidf_normalized(self, tfidf, cve):
        self.nb_pipeline_tfidf_normalized = Pipeline([
            ('NBCV', tfidf.tfidf_vectorizer_normalized),
            ('nb_clf', MultinomialNB())])

        self.nb_pipeline_tfidf_normalized.fit(cve.df_normalized['articleBody'],
                                              cve.df['Stance'])
        self.predicted_nb_tfidf_normalized = self.nb_pipeline_tfidf.predict(
            CsvReader.test_df_normalized['articleBody'])
        self.nb_mean_tfidf_normalized = numpy.mean(self.predicted_nb_tfidf_normalized == CsvReader.test_df['Stance'])
        print('Naive Bayes with TFIDF and Normalized', self.nb_mean_tfidf_normalized)

    def build_naive_bayes_ngram_cv(self, cve):
        self.nb_pipeline_ngram_cv = Pipeline([
            ('nb_tfidf', cve.count_vectorizer_ngram),
            ('nb_clf', MultinomialNB())])

        self.nb_pipeline_ngram_cv.fit(cve.df['articleBody'], cve.df['Stance'])
        predicted_nb_ngram_cv = self.nb_pipeline_ngram_cv.predict(
            CsvReader.test_df['articleBody'])
        self.nb_mean_ngram_cv = numpy.mean(predicted_nb_ngram_cv == cve.df['Stance'])
        print('Naive Bayes with Count Vector N-Grams', self.nb_mean_ngram_cv)

    def build_naive_bayes_ngram_tfidf(self, tfidf, cve):
        self.nb_pipeline_ngram_tfidf = Pipeline([
            ('nb_tfidf', tfidf.tfidf_vectorizer_ngram),
            ('nb_clf', MultinomialNB())])

        self.nb_pipeline_ngram_tfidf.fit(cve.df['articleBody'], cve.df['Stance'])
        predicted_nb_ngram_tfidf = self.nb_pipeline_ngram_tfidf.predict(
            CsvReader.test_df['articleBody'])
        self.nb_mean_ngram_tfidf = numpy.mean(predicted_nb_ngram_tfidf == cve.df['Stance'])
        print('Naive Bayes with TFIDF N-Grams', self.nb_mean_ngram_tfidf)

    def build_naive_bayes_ngram_cv_normalized(self, cve):
        self.nb_pipeline_ngram_cv_normalized = Pipeline([
            ('nb_tfidf', cve.count_vectorizer_ngram),
            ('nb_clf', MultinomialNB())])

        self.nb_pipeline_ngram_cv_normalized.fit(cve.df_normalized['articleBody'],
                                                 cve.df['Stance'])
        predicted_nb_ngram_cv_normalized = self.nb_pipeline_ngram_cv_normalized.predict(
            CsvReader.test_df_normalized['articleBody'])
        self.nb_mean_ngram_cv_normalized = numpy.mean(predicted_nb_ngram_cv_normalized == CsvReader.test_df['Stance'])
        print('Naive Bayes with Count Vector N-Grams normalized', self.nb_mean_ngram_cv_normalized)

    def build_naive_bayes_ngram_tfidf_normalized(self, tfidf, cve):
        self.nb_pipeline_ngram_tfidf_normalized = Pipeline([
            ('nb_tfidf_normalized', tfidf.tfidf_vectorizer_ngram),
            ('nb_clf', MultinomialNB())])
        self.nb_pipeline_ngram_tfidf_normalized.fit(cve.df_normalized['articleBody'],
                                                    cve.df['Stance'])
        predicted_nb_ngram_tfidf_normalized = self.nb_pipeline_ngram_tfidf_normalized.predict(
            CsvReader.test_df_normalized['articleBody'])
        self.nb_mean_ngram_tfidf_normalized = numpy.mean(
            predicted_nb_ngram_tfidf_normalized == CsvReader.test_df['Stance'])
        print('Naive Bayes with TFIDF N-Grams, normalized', self.nb_mean_ngram_tfidf_normalized)

    def build_naive_bayes_pos_tag_cv(self, cve):
        self.nb_pipeline_pos_tag_cv = Pipeline([
            ('nb_tfidf', cve.count_vectorizer_ngram),
            ('nb_clf', MultinomialNB())])

        self.nb_pipeline_pos_tag_cv.fit(CsvReader.train_pos_tag['articleBody'],
                                        cve.df['Stance'])
        predicted_nb_pos_tag_cv = self.nb_pipeline_pos_tag_cv.predict(
            CsvReader.test_pos_tag['articleBody'])
        self.nb_mean_pos_tag_cv = numpy.mean(predicted_nb_pos_tag_cv == CsvReader.test_df['Stance'])
        print('Naive Bayes with Count Vector with POS tagging', self.nb_mean_pos_tag_cv)

    def build_naive_bayes_pos_tag_tfidf(self, tfidf, cve):
        self.nb_pipeline_pos_tag_tfidf = Pipeline([
            ('nb_tfidf', tfidf.tfidf_vectorizer_pos_tag),
            ('nb_clf', MultinomialNB())])

        self.nb_pipeline_pos_tag_tfidf.fit(CsvReader.train_pos_tag['articleBody'],
                                           cve.df['Stance'])
        predicted_nb_pos_tag_tfidf = self.nb_pipeline_pos_tag_tfidf.predict(
            CsvReader.test_pos_tag['articleBody'])
        self.nb_mean_pos_tag_tfidf = numpy.mean(predicted_nb_pos_tag_tfidf == CsvReader.test_df['Stance'])
        print('Naive Bayes with TFIDF with POS tagging', self.nb_mean_pos_tag_tfidf)

    def __init__(self, cve, tfidfe):
        self.build_naive_bayes_classifier_cv(cve)
        # self.build_naive_bayes_classifier_tfidf(tfidfe, cve)
        self.build_naive_bayes_classifier_cv_normalized(cve)
        # self.build_naive_bayes_classifier_tfidf_normalized(tfidfe, cve)
        self.build_naive_bayes_ngram_cv(cve)
        self.build_naive_bayes_ngram_cv_normalized(cve)
        # self.build_naive_bayes_ngram_tfidf(tfidfe, cve)
        # self.build_naive_bayes_ngram_tfidf_normalized(tfidfe, cve)
        self.build_naive_bayes_pos_tag_cv(cve)
        # self.build_naive_bayes_pos_tag_tfidf(tfidfe, cve)


if __name__ == '__main__':
    cve = CountVectorizerExtractor()
    tve = TfIdfVectorizerExtractor(cve)
    nb_classifier = NaiveBayesClassifier(cve=cve, tfidfe=tve)
