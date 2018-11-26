import re
import unicodedata

import nltk
import numpy
import pandas
import spacy
from bs4 import BeautifulSoup
from contractions import contractions_dict

nlp = spacy.load('en_core_web_sm')

# from CountVectorizerExtraction import CountVectorizerExtractor
from nltk.tokenize.toktok import ToktokTokenizer

# nltk.download('stopwords')
tokenizer = ToktokTokenizer()
# tweet_tokenizer = nltk.TweetTokenizer()

stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')
stopword_list.append('id')


def strip_html_tags(soup):
    stripped_text = soup.get_text()
    return stripped_text


#
# def stem_tokens(tokens, stemmer):
#     stemmed = []
#     for token in tokens:
#         stemmed.append(stemmer.stem(token))
#     return stemmed

#
# def porter_stem_dataframe_tokens(dataframe):
#     porter = nltk.PorterStemmer()
#     stem_df = dataframe.copy()
#     for index, row in stem_df.iterrows():
#         row['Headline'] = stem_tokens(tokenizer.tokenize(row['Headline']), porter)
#         row['articleBody'] = stem_tokens(tokenizer.tokenize(row['articleBody']), porter)
#     return stem_df


# def porter_stem_df(dataframe):
#     porter = nltk.PorterStemmer()
#     stem_df = dataframe.copy()
#     for index, row in stem_df.iterrows():
#         if index < 5:
#             headline = row['Headline']
#             for word in headline.split():
#                 row['Headline'] = row['Headline'].replace(word, porter.stem(word))
#             body = row['articleBody']
#             for word in body.split():
#                 row['articleBody'] = row['articleBody'].replace(word, porter.stem(word))
#     return stem_df


# def snowball_stem_dataframe(dataframe):
#     sb_stemmer = nltk.SnowballStemmer('english')
#     stem_df = dataframe.copy()
#     for index, row in stem_df.iterrows():
#         row['Headline'] = stem_tokens(row['Headline'], sb_stemmer)
#         row['articleBody'] = stem_tokens(row['articleBody'], sb_stemmer)
#     return stem_df


# def tokenize_dataframe(dataframe):
#     columns = ['Body ID', 'Headline', 'articleBody', 'Stance']
#     tokenized_dataframe = pd.DataFrame(columns=columns)
#     for index, row in dataframe.iterrows():
#         if index < 5:
#             tokenized_row = row.copy()
#             headline = tokenizer.tokenize(row['Headline'])
#             tokenized_row['Headline'] = [word.lower() for word in headline if
#                                          word not in stopword_list and len(word) > 1]
#             body = tokenizer.tokenize(row['articleBody'])
#             tokenized_row['articleBody'] = [word.lower() for word in body if
#                                             word not in stopword_list and len(word) > 1]
#             tokenized_dataframe.loc[index] = tokenized_row
#     print(tokenized_dataframe)
#     return tokenized_dataframe
def tokenize(text):
    return tokenizer.tokenize(text)


def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def expand_contractions(text, contraction_mapping=contractions_dict):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=nltk.re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        if expanded_contraction is not None:
            expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def porter_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text


def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text


def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text


def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def normalize_string(text, contraction_expansion=True,
                     accented_char_removal=True, text_lower_case=True,
                     text_lemmatization=True, special_char_removal=True,
                     stopword_removal=True, remove_digits=True):
    if accented_char_removal:
        text = remove_accented_chars(text)
        # expand contractions
    if contraction_expansion:
        text = expand_contractions(text)
        # lowercase the text
    if text_lower_case:
        text = text.lower()
        # remove extra newlines
    text = re.sub(r'[\r|\n|\r\n]+', ' ', text)
    # lemmatize text
    if text_lemmatization:
        text = lemmatize_text(text)
    # remove special characters and\or digits
    if special_char_removal:
        # insert spaces between special characters to isolate them
        special_char_pattern = re.compile(r'([{.(-)!}])')
        text = special_char_pattern.sub(" \\1 ", text)
        text = remove_special_characters(text, remove_digits=remove_digits)
        # remove extra whitespace
    text = re.sub(' +', ' ', text)
    # remove stopwords
    if stopword_removal:
        text = remove_stopwords(text, is_lower_case=text_lower_case)
    return text


def normalize_df(df, contraction_expansion=True,
                 accented_char_removal=True, text_lower_case=True,
                 text_lemmatization=True, special_char_removal=True,
                 stopword_removal=True, remove_digits=True):
    # normalized_df = df.copy()
    normalized_df = pandas.DataFrame(columns=['Body ID', 'Headline', 'articleBody'])
    for index, row in df.iterrows():
        # print(index)
        body_id = row['Body ID']
        headline = row['Headline']
        article_body = row['articleBody']
        if text_lower_case:
            headline = headline.lower()
            article_body = article_body.lower()

        if accented_char_removal:
            headline = remove_accented_chars(headline)
            article_body = remove_accented_chars(article_body)
        if contraction_expansion:
            headline = expand_contractions(headline)
            article_body = expand_contractions(article_body)

        headline = re.sub(r'[\r|\n|\r\n]+', ' ', headline)
        article_body = re.sub(r'[\r|\n|\r\n]+', ' ', article_body)
        # lemmatize text
        if text_lemmatization:
            headline = lemmatize_text(headline)
            article_body = lemmatize_text(article_body)
        else:
            headline = porter_stemmer(headline)
            article_body = porter_stemmer(article_body)
        # remove special characters and\or digits
        if special_char_removal:
            # insert spaces between special characters to isolate them
            special_char_pattern = re.compile(r'([{.(-)!}])')
            headline = special_char_pattern.sub(" \\1 ", headline)
            headline = remove_special_characters(headline, remove_digits=remove_digits)
            article_body = special_char_pattern.sub(" \\1 ",
                                                    article_body)
            article_body = remove_special_characters(article_body,
                                                     remove_digits=remove_digits)
            # remove extra whitespace
            headline = re.sub(' +', ' ', headline)
            article_body = re.sub(' +', ' ', article_body)
        # remove stopwords
        if stopword_removal:
            headline = remove_stopwords(headline, is_lower_case=text_lower_case)
            article_body = remove_stopwords(article_body, is_lower_case=text_lower_case)
        # print(headline)
        # print(article_body)
        normalized_df.loc[index] = [body_id, headline, article_body]
    return normalized_df


def pos_tagging(df):
    tagged_df = pandas.DataFrame(columns=['Body ID', 'Headline', 'articleBody'])
    for index, row in df.iterrows():
        headline = str(row.Headline)
        headline_nlp = nlp(headline)
        body = str(row.articleBody)
        body_nlp = nlp(body)
        pos_tagged_headline = [(word, word.tag_, word.pos_) for word in headline_nlp]
        pos_tagged_body = [(word, word.tag_, word.pos_) for word in body_nlp]
        tagged_df.loc[index] = [row['Body ID'], pos_tagged_headline, pos_tagged_body]

    # tagged_df = pandas.DataFrame(pos_tagged, columns=['Word', 'POS tag', 'Tag Type'])
    # print(tagged_df)
    return tagged_df


def tagged_df_to_vector(df):
    t2v = {}
    for index, row in df.iterrows():
        t2v = {**t2v, **{tag[0]: numpy.array([tag[1], tag[2]]) for tag in row['Headline'] + row['articleBody']}}
        # merge_two_dicts(t2v,)
    return t2v

# if __name__ == '__main__':
# df = CsvReader.test_df
# normalized_df = CsvReader.train_df_normalized
# CsvReader.data_frame_to_csv(normalized_df, './test_normalized.csv')
# # normalized_df = CsvReader.train_df_normalized
# # CsvReader.data_frame_to_csv(normalized_df, './normalized.csv')
# normalized_pos_tag_df = pos_tagging(normalized_df)
# CsvReader.data_frame_to_csv(normalized_pos_tag_df, './train_tagged.csv')

# print(normalized_pos_tag_df)
# t2v = tagged_df_to_vector(normalized_pos_tag_df)
# print(t2v)
# CsvReader.data_frame_to_csv(normalized_pos_tag_df, './normalized_postag.csv')
