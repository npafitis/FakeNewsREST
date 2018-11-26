import pandas as pd

# Read and merge by Body ID csv files.


# pd.set_option('display.max_columns', 500)


def read_dataframe_train(stances_file, body_file):
    train_articles_stances = pd.read_csv(stances_file)
    train_articles_bodies = pd.read_csv(body_file)
    train_articles = pd.merge(train_articles_stances,
                              train_articles_bodies,
                              left_on='Body ID',
                              right_on='Body ID',
                              how='left')
    return train_articles


def read_normalized_df(filename):
    return pd.read_csv(filename)


def test_dataframe(train_articles):
    print("Dataframe size:")
    print(train_articles.shape)
    print(train_articles.head(5))


# For testing
def data_frame_to_csv(dataframe, path):
    dataframe.to_csv(path, sep=',')


train_df = read_dataframe_train('./resources/train_stances.csv', './resources/train_bodies.csv').head(n=500)
# train_df_normalized = porter_stem_dataframe_tokens(train_df)
# for index, row in train_df.iterrows():
#     row['Headline'] = DataFormatter.remove_special_characters(row['Headline'])
#     row['articleBody'] = DataFormatter.remove_special_characters(row['articleBody'])
train_df_normalized = read_normalized_df('./normalized.csv')
data_frame_to_csv(train_df_normalized, './normalized.csv')
test_df = read_dataframe_train('./resources/competition_test_stances.csv',
                               './resources/competition_test_bodies.csv').head(n=500)
# for index, row in test_df.iterrows():
#     row['Headline'] = DataFormatter.remove_special_characters(row['Headline'])
#     row['articleBody'] = DataFormatter.remove_special_characters(row['articleBody'])
# test_df_normalized = DataFormatter.normalize_df(test_df)
test_df_normalized = read_normalized_df('./normalized_test.csv')
data_frame_to_csv(test_df_normalized, './normalized_test.csv')
# train_pos_tag = pos_tagging(train_df_normalized)
# test_pos_tag = pos_tagging(test_df_normalized)

# train_pos_tag = pos_tagging(train_df_normalized)
train_pos_tag = read_normalized_df('./train_pos_tag.csv')
# test_pos_tag = pos_tagging(test_df_normalized)
test_pos_tag = read_normalized_df('./test_pos_tag.csv')

data_frame_to_csv(train_pos_tag, './train_pos_tag.csv')

data_frame_to_csv(test_pos_tag, './test_pos_tag.csv')

# For testing


if __name__ == '__main__':
    # print(test_df)
    # data_frame_to_csv(train_pos_tag, './train_pos_tag.csv')
    # data_frame_to_csv(test_pos_tag, './test_pos_tag.csv')
    # test_dataframe(test_df)
    print(train_pos_tag)
    print(train_pos_tag.columns)
    # print(train_df_normalized.head())
