from sklearn.feature_extraction.text import CountVectorizer

import CsvReader


class CountVectorizerExtractor:
    count_vectorizer = CountVectorizer(min_df=1)
    count_vectorizer_normalized = CountVectorizer(min_df=1)
    count_vectorizer_ngram = CountVectorizer(ngram_range=(1, 3), stop_words='english', min_df=1)
    count_vectorizer_ngram_normalized = CountVectorizer(ngram_range=(1, 3), stop_words='english', min_df=1)
    count_vectorizer_pos_tag = CountVectorizer(min_df=1)

    df = CsvReader.train_df
    df_normalized = CsvReader.train_df_normalized
    td_matrix = {}
    td_matrix_ngram = {}
    td_matrix_normalized = {}
    td_matrix_ngram_normalized = {}
    td_matrix_pos_tag = {}

    def calc_count_vectorizer_pos_tag(self):
        print('Calculating Term-Document Matrix for Part-of-Speech tagged words')
        td_matrix_pos_tag = self.count_vectorizer_pos_tag.fit_transform(
            CsvReader.train_pos_tag
        )
        print('Term-Document Matrix for Part-of-Speech tagged words calculated')
        return td_matrix_pos_tag

    def calc_count_vectorizer_raw(self):
        print('Calculating Term-Document Matrix')
        td_matrix = self.count_vectorizer.fit_transform((self.df['Headline'] + self.df['articleBody']).values)
        print('Term-Document Matrix calculated')
        return td_matrix

    def calc_count_vectorizer_normalized(self):
        print('Calculating Normalized Term-Document Matrix')
        td_matrix_normalized = self.count_vectorizer_normalized.fit_transform(
            (self.df_normalized['Headline'] + self.df_normalized['articleBody']).values)
        print('Term-Document Normalized Matrix calculated')
        return td_matrix_normalized

    def calc_count_vectorizer_ngram(self):
        print('Calculating Term-Document Matrix for ngrams')
        td_matrix_ngram = self.count_vectorizer_ngram.fit_transform(
            (self.df['Headline'] + self.df['articleBody']).values)
        print('Term-Document Matrix for ngrams calculated')
        return td_matrix_ngram

    def calc_count_vectorizer_ngram_normalized(self):
        print('Calculating Term-Document Matrix for ngrams normalized')
        td_matrix_ngram = self.count_vectorizer_ngram_normalized.fit_transform(
            (self.df_normalized['Headline'] + self.df_normalized['articleBody']).values)
        print('Calculating Term-Document Matrix for ngrams normalized')
        return td_matrix_ngram

    def __init__(self):
        self.td_matrix = self.calc_count_vectorizer_raw()
        self.td_matrix_normalized = self.calc_count_vectorizer_normalized()
        self.td_matrix_ngram = self.calc_count_vectorizer_ngram()
        self.td_matrix_ngram_normalized = self.calc_count_vectorizer_ngram_normalized()
        self.td_matrix_pos_tag = self.calc_count_vectorizer_pos_tag()

    # For testing
    def print_dataframe(self):
        print(self.df.shape)
        print(self.df[:5])
        print(self.df[:5].values)

    # For testing
    def print_vocabulary(self):
        # print(self.count_vectorizer.vocabulary_)
        print(self.count_vectorizer_normalized.vocabulary_)
        # print(self.count_vectorizer_ngram.vocabulary_)
        # print(self.count_vectorizer_ngram_normalized.vocabulary_)
        # print(self.count_vectorizer_pos_tag.vocabulary_)

    def print_td_matrix(self):
        # Print td-matrix shape
        print(self.td_matrix.shape)

        # get feature names
        print(self.count_vectorizer.get_feature_names()[:25])


if __name__ == '__main__':
    cve = CountVectorizerExtractor()
    cve.print_vocabulary()
    # headline = cve.dataframe_normalized.loc[1]['Headline']
    # body = cve.dataframe_normalized.loc[1]['articleBody']
    # print(headline)
    # print(body)
    # print(jaccard_similarity(headline, body))
    # print(cosine_similarity(headline, body))
    # headline = cve.dataframe.loc[1]['Headline']
    # body = cve.dataframe.loc[1]['articleBody']
    # print(cosine_similarity(headline, body))
    # cve.print_dataframe()
    # cve.print_vocabulary()
    # cve.print_td_matrix()
