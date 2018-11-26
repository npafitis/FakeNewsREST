# string to test
import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

import CsvReader
from CountVectorizerExtraction import CountVectorizerExtractor
from TfIdfVectorizerExtraction import TfIdfVectorizerExtractor


class RandomForestClassification:

    def build_rf_classifier_cv(self, cve):
        self.rf_pipeline_cv = Pipeline([
            ('RFCV', cve.count_vectorizer),
            ('rf_clf', RandomForestClassifier(n_estimators=200, n_jobs=3))])

        self.rf_pipeline_cv.fit(cve.df['articleBody'], cve.df['Stance'])
        self.predicted_rf_cv = self.rf_pipeline_cv.predict(
            CsvReader.test_df['articleBody'])
        self.rf_mean_cv = numpy.mean(self.predicted_rf_cv == cve.df['Stance'])
        print('Random Forest with Count Vector', self.rf_mean_cv)

    def build_rf_classifier_tfidf(self, tfidf, cve):
        self.rf_pipeline_tfidf = Pipeline([
            ('RFCV', tfidf.tfidf_vectorizer),
            ('rf_clf', RandomForestClassifier(n_estimators=200, n_jobs=3))])

        self.rf_pipeline_tfidf.fit(cve.df['articleBody'], cve.df['Stance'])
        self.predicted_rf_tfidf = self.rf_pipeline_tfidf.predict(
            CsvReader.test_df['articleBody'])
        self.rf_mean_tfidf = numpy.mean(self.predicted_rf_tfidf == cve.df['Stance'])
        print('Random Forest with TFIDF', self.rf_mean_tfidf)

    def build_rf_classifier_cv_normalized(self, cve):
        self.rf_pipeline_cv_normalized = Pipeline([
            ('RFCV', cve.count_vectorizer_normalized),
            ('rf_clf', RandomForestClassifier(n_estimators=200, n_jobs=3))])

        self.rf_pipeline_cv_normalized.fit(cve.df_normalized['articleBody'],
                                           cve.df['Stance'])
        self.predicted_rf_cv_normalized = self.rf_pipeline_cv_normalized.predict(
            CsvReader.test_df_normalized['articleBody'])
        self.rf_mean_cv_normalized = numpy.mean(self.predicted_rf_cv_normalized == CsvReader.test_df['Stance'])
        print('Random Forest with Count Vector and normalized', self.rf_mean_cv_normalized)

    def build_rf_classifier_tfidf_normalized(self, tfidf, cve):
        self.rf_pipeline_tfidf_normalized = Pipeline([
            ('RFCV', tfidf.tfidf_vectorizer_normalized),
            ('rf_clf', RandomForestClassifier(n_estimators=200, n_jobs=3))])

        self.rf_pipeline_tfidf_normalized.fit(cve.df_normalized['articleBody'],
                                              cve.df['Stance'])
        self.predicted_rf_tfidf_normalized = self.rf_pipeline_tfidf.predict(
            CsvReader.test_df_normalized['articleBody'])
        self.rf_mean_tfidf_normalized = numpy.mean(self.predicted_rf_tfidf_normalized == CsvReader.test_df['Stance'])
        print('Random Forest with TFIDF and Normalized', self.rf_mean_tfidf_normalized)

    def build_rf_ngram_cv(self, cve):
        self.rf_pipeline_ngram_cv = Pipeline([
            ('rf_tfidf', cve.count_vectorizer_ngram),
            ('rf_clf', RandomForestClassifier(n_estimators=200, n_jobs=3))])

        self.rf_pipeline_ngram_cv.fit(cve.df['articleBody'], cve.df['Stance'])
        predicted_rf_ngram_cv = self.rf_pipeline_ngram_cv.predict(
            CsvReader.test_df['articleBody'])
        self.rf_mean_ngram_cv = numpy.mean(predicted_rf_ngram_cv == cve.df['Stance'])
        print('Random Forest with Count Vector N-Grams', self.rf_mean_ngram_cv)

    def build_rf_ngram_tfidf(self, tfidf, cve):
        self.rf_pipeline_ngram_tfidf = Pipeline([
            ('rf_tfidf', tfidf.tfidf_vectorizer_ngram),
            ('rf_clf', RandomForestClassifier(n_estimators=200, n_jobs=3))])

        self.rf_pipeline_ngram_tfidf.fit(cve.df['articleBody'], cve.df['Stance'])
        predicted_rf_ngram_tfidf = self.rf_pipeline_ngram_tfidf.predict(
            CsvReader.test_df['articleBody'])
        self.rf_mean_ngram_tfidf = numpy.mean(predicted_rf_ngram_tfidf == cve.df['Stance'])
        print('Random Forest with TFIDF N-Grams', self.rf_mean_ngram_tfidf)

    def build_rf_ngram_cv_normalized(self, cve):
        self.rf_pipeline_ngram_cv_normalized = Pipeline([
            ('rf_tfidf', cve.count_vectorizer_ngram),
            ('rf_clf', RandomForestClassifier(n_estimators=200, n_jobs=3))])

        self.rf_pipeline_ngram_cv_normalized.fit(numpy.array(cve.df_normalized['articleBody']),
                                                 numpy.array(cve.df['Stance']))
        predicted_rf_ngram_cv_normalized = self.rf_pipeline_ngram_cv_normalized.predict(
            numpy.array(CsvReader.test_df_normalized['articleBody']))
        self.rf_mean_ngram_cv_normalized = numpy.mean(
            predicted_rf_ngram_cv_normalized == CsvReader.test_df['Stance'])
        print('Random Forest with Count Vector N-Grams normalised', self.rf_mean_ngram_cv_normalized)

    def build_rf_ngram_tfidf_normalized(self, tfidf, cve):
        self.rf_pipeline_ngram_tfidf_normalized = Pipeline([
            ('rf_tfidf_normalized', tfidf.tfidf_vectorizer_ngram),
            ('rf_clf', RandomForestClassifier(n_estimators=200, n_jobs=3))])
        self.rf_pipeline_ngram_tfidf_normalized.fit(
            cve.df_normalized['articleBody'],
            cve.df['Stance'])
        predicted_rf_ngram_tfidf_normalized = self.rf_pipeline_ngram_tfidf_normalized.predict(
            CsvReader.test_df_normalized['articleBody'])
        self.rf_mean_ngram_tfidf_normalized = numpy.mean(
            predicted_rf_ngram_tfidf_normalized == CsvReader.test_df['Stance'])
        print('Random Forest with TFIDF N-Grams, normalized', self.rf_mean_ngram_tfidf_normalized)

    def build_rf_pos_tag_cv(self, cve):
        self.rf_pipeline_pos_tag_cv = Pipeline([
            ('rf_tfidf', cve.count_vectorizer_ngram),
            ('rf_clf', RandomForestClassifier(n_estimators=200, n_jobs=3))])

        self.rf_pipeline_pos_tag_cv.fit(CsvReader.train_pos_tag['articleBody'],
                                        cve.df['Stance'])
        predicted_rf_pos_tag_cv = self.rf_pipeline_pos_tag_cv.predict(
            CsvReader.test_pos_tag['articleBody'])
        self.rf_mean_pos_tag_cv = numpy.mean(predicted_rf_pos_tag_cv == CsvReader.test_df['Stance'])
        print('Random Forest with Count Vector with POS tagging', self.rf_mean_pos_tag_cv)

    def build_rf_pos_tag_tfidf(self, tfidf, cve):
        self.rf_pipeline_pos_tag_tfidf = Pipeline([
            ('rf_tfidf', tfidf.tfidf_vectorizer_pos_tag),
            ('rf_clf', RandomForestClassifier(n_estimators=200, n_jobs=3))])

        self.rf_pipeline_pos_tag_tfidf.fit(CsvReader.train_pos_tag['articleBody'],
                                           cve.df['Stance'])
        predicted_rf_pos_tag_tfidf = self.rf_pipeline_pos_tag_tfidf.predict(CsvReader.test_pos_tag['articleBody'])
        self.rf_mean_pos_tag_tfidf = numpy.mean(predicted_rf_pos_tag_tfidf == CsvReader.test_df['Stance'])
        print('Random Forest with TFIDF with POS tagging', self.rf_mean_pos_tag_tfidf)

    def __init__(self, cve, tfidfe):
        self.build_rf_classifier_cv(cve)
        # self.build_rf_classifier_tfidf(tfidfe, cve)
        self.build_rf_classifier_cv_normalized(cve)
        # self.build_rf_classifier_tfidf_normalized(tfidfe, cve)
        self.build_rf_ngram_cv(cve)
        self.build_rf_ngram_cv_normalized(cve)
        # self.build_rf_ngram_tfidf(tfidfe, cve)
        # self.build_rf_ngram_tfidf_normalized(tfidfe, cve)
        self.build_rf_pos_tag_cv(cve)
        # self.build_rf_pos_tag_tfidf(tfidfe, cve)


if __name__ == '__main__':
    cve = CountVectorizerExtractor()
    tve = TfIdfVectorizerExtractor(cve)
    rf_classifier = RandomForestClassification(cve=cve, tfidfe=tve)
