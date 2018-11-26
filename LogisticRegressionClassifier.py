# string to test
import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import CsvReader
from CountVectorizerExtraction import CountVectorizerExtractor
from TfIdfVectorizerExtraction import TfIdfVectorizerExtractor


class LogisticRegressionClassifier:

    def build_lr_classifier_cv(self, cve):
        self.lr_pipeline_cv = Pipeline([
            ('LRCV', cve.count_vectorizer),
            ('lr_clf', LogisticRegression())])

        self.lr_pipeline_cv.fit(cve.df['Headline'] + cve.df['articleBody'], cve.df['Stance'])
        self.predicted_lr_cv = self.lr_pipeline_cv.predict(
            CsvReader.test_df['Headline'] + CsvReader.test_df['articleBody'])
        self.lr_mean_cv = numpy.mean(self.predicted_lr_cv == cve.df['Stance'])
        print('Logistic Regression with Count Vector',self.lr_mean_cv)

    def build_lr_classifier_tfidf(self, tfidf, cve):
        self.lr_pipeline_tfidf = Pipeline([
            ('LRCV', tfidf.tfidf_vectorizer),
            ('lr_clf', LogisticRegression())])

        self.lr_pipeline_tfidf.fit(cve.df['Headline'] + cve.df['articleBody'], cve.df['Stance'])
        self.predicted_lr_tfidf = self.lr_pipeline_tfidf.predict(
            CsvReader.test_df['Headline'] + CsvReader.test_df['articleBody'])
        self.lr_mean_tfidf = numpy.mean(self.predicted_lr_tfidf == cve.df['Stance'])
        print('Logistic Regression with TFIDF',self.lr_mean_tfidf)

    def build_lr_classifier_cv_normalized(self, cve):
        self.lr_pipeline_cv_normalized = Pipeline([
            ('LRCV', cve.count_vectorizer_normalized),
            ('lr_clf', LogisticRegression())])

        self.lr_pipeline_cv_normalized.fit(cve.df_normalized['Headline'] + cve.df_normalized['articleBody'],
                                           cve.df['Stance'])
        self.predicted_lr_cv_normalized = self.lr_pipeline_cv_normalized.predict(
            CsvReader.test_df_normalized['Headline'] + CsvReader.test_df_normalized['articleBody'])
        self.lr_mean_cv_normalized = numpy.mean(self.predicted_lr_cv_normalized == CsvReader.test_df['Stance'])
        print('Logistic Regression with Count Vector and normalized',self.lr_mean_cv_normalized)

    def build_lr_classifier_tfidf_normalized(self, tfidf, cve):
        self.lr_pipeline_tfidf_normalized = Pipeline([
            ('LRCV', tfidf.tfidf_vectorizer_normalized),
            ('lr_clf', LogisticRegression())])

        self.lr_pipeline_tfidf_normalized.fit(cve.df_normalized['Headline'] + cve.df_normalized['articleBody'],
                                              cve.df['Stance'])
        self.predicted_lr_tfidf_normalized = self.lr_pipeline_tfidf.predict(
            CsvReader.test_df_normalized['Headline'] + CsvReader.test_df_normalized['articleBody'])
        self.lr_mean_tfidf_normalized = numpy.mean(self.predicted_lr_tfidf_normalized == CsvReader.test_df['Stance'])
        print('Logistic Regression with TFIDF and Normalized',self.lr_mean_tfidf_normalized)

    def build_lr_ngram_cv(self, cve):
        self.lr_pipeline_ngram_cv = Pipeline([
            ('lr_tfidf', cve.count_vectorizer_ngram),
            ('lr_clf', LogisticRegression())])

        self.lr_pipeline_ngram_cv.fit(cve.df['Headline'] + cve.df['articleBody'], cve.df['Stance'])
        predicted_lr_ngram_cv = self.lr_pipeline_ngram_cv.predict(
            CsvReader.test_df['Headline'] + CsvReader.test_df['articleBody'])
        self.lr_mean_ngram_cv = numpy.mean(predicted_lr_ngram_cv == cve.df['Stance'])
        print('Logistic Regression with Count Vector N-Grams',self.lr_mean_ngram_cv)

    def build_lr_ngram_tfidf(self, tfidf, cve):
        self.lr_pipeline_ngram_tfidf = Pipeline([
            ('lr_tfidf', tfidf.tfidf_vectorizer_ngram),
            ('lr_clf', LogisticRegression())])

        self.lr_pipeline_ngram_tfidf.fit(cve.df['Headline'] + cve.df['articleBody'], cve.df['Stance'])
        predicted_lr_ngram_tfidf = self.lr_pipeline_ngram_tfidf.predict(
            CsvReader.test_df['Headline'] + CsvReader.test_df['articleBody'])
        self.lr_mean_ngram_tfidf = numpy.mean(predicted_lr_ngram_tfidf == cve.df['Stance'])
        print('Logistic Regression with TFIDF N-Grams',self.lr_mean_ngram_tfidf)

    def build_lr_ngram_cv_normalized(self, cve):
        self.lr_pipeline_ngram_cv_normalized = Pipeline([
            ('lr_tfidf', cve.count_vectorizer_ngram),
            ('lr_clf', LogisticRegression())])

        self.lr_pipeline_ngram_cv_normalized.fit(cve.df_normalized['Headline'] + cve.df_normalized['articleBody'],
                                                 cve.df['Stance'])
        predicted_lr_ngram_cv_normalized = self.lr_pipeline_ngram_cv_normalized.predict(
            CsvReader.test_df_normalized['Headline'] + CsvReader.test_df_normalized['articleBody'])
        self.lr_mean_ngram_cv_normalized = numpy.mean(predicted_lr_ngram_cv_normalized == CsvReader.test_df['Stance'])
        print('Logistic Regression with Count Vector N-Grams normalized',self.lr_mean_ngram_cv_normalized)

    def build_lr_ngram_tfidf_normalized(self, tfidf, cve):
        self.lr_pipeline_ngram_tfidf_normalized = Pipeline([
            ('lr_tfidf_normalized', tfidf.tfidf_vectorizer_ngram),
            ('lr_clf', LogisticRegression())])
        self.lr_pipeline_ngram_tfidf_normalized.fit(cve.df_normalized['Headline'] + cve.df_normalized['articleBody'],
                                                    cve.df['Stance'])
        predicted_lr_ngram_tfidf_normalized = self.lr_pipeline_ngram_tfidf_normalized.predict(
            CsvReader.test_df_normalized['Headline'] + CsvReader.test_df_normalized['articleBody'])
        self.lr_mean_ngram_tfidf_normalized = numpy.mean(
            predicted_lr_ngram_tfidf_normalized == CsvReader.test_df['Stance'])
        print('Logistic Regression with TFIDF N-Grams, normalized',self.lr_mean_ngram_tfidf_normalized)

    def build_lr_pos_tag_cv(self, cve):
        self.lr_pipeline_pos_tag_cv = Pipeline([
            ('lr_tfidf', cve.count_vectorizer_ngram),
            ('lr_clf', LogisticRegression())])

        self.lr_pipeline_pos_tag_cv.fit(CsvReader.train_pos_tag['Headline']+CsvReader.train_pos_tag['articleBody'],
                                        cve.df['Stance'])
        predicted_lr_pos_tag_cv = self.lr_pipeline_pos_tag_cv.predict(
            CsvReader.test_pos_tag['Headline'] + CsvReader.test_pos_tag['articleBody'])
        self.lr_mean_pos_tag_cv = numpy.mean(predicted_lr_pos_tag_cv == CsvReader.test_df['Stance'])
        print('Logistic Regression with Count Vector with POS tagging',self.lr_mean_pos_tag_cv)

    def build_lr_pos_tag_tfidf(self, tfidf, cve):
        self.lr_pipeline_pos_tag_tfidf = Pipeline([
            ('lr_tfidf', tfidf.tfidf_vectorizer_pos_tag),
            ('lr_clf', LogisticRegression())])

        self.lr_pipeline_pos_tag_tfidf.fit(CsvReader.train_pos_tag,
                                           cve.df['Stance'])
        predicted_lr_pos_tag_tfidf = self.lr_pipeline_pos_tag_tfidf.predict(
            CsvReader.test_pos_tag)
        self.lr_mean_pos_tag_tfidf = numpy.mean(predicted_lr_pos_tag_tfidf == CsvReader.test_df['Stance'])
        print('Logistic Regression with TFIDF with POS tagging',self.lr_mean_pos_tag_tfidf)

    def __init__(self, cve, tfidfe):
        self.build_lr_classifier_cv(cve)
        # self.build_lr_classifier_tfidf(tfidfe, cve)
        self.build_lr_classifier_cv_normalized(cve)
        # self.build_lr_classifier_tfidf_normalized(tfidfe, cve)
        self.build_lr_ngram_cv(cve)
        self.build_lr_ngram_cv_normalized(cve)
        # self.build_lr_ngram_tfidf(tfidfe, cve)
        # self.build_lr_ngram_tfidf_normalized(tfidfe, cve)
        self.build_lr_pos_tag_cv(cve)
        # self.build_lr_pos_tag_tfidf(tfidfe, cve)


if __name__ == '__main__':
    cve = CountVectorizerExtractor()
    tve = TfIdfVectorizerExtractor(cve)
    lr_classifier = LogisticRegressionClassifier(cve=cve, tfidfe=tve)
