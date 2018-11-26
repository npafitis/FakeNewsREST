import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, learning_curve

from CountVectorizerExtraction import CountVectorizerExtractor
from LogisticRegressionClassifier import LogisticRegressionClassifier
from NaiveBayesClassifier import NaiveBayesClassifier
from RandomForestClassification import RandomForestClassification
from TfIdfVectorizerExtraction import TfIdfVectorizerExtractor


def plot_learing_curve(pipeline, title, x, y):
    size = 500
    cv = KFold(size, shuffle=True)

    # x = x
    # y = DataPrep.train_news["Label"]

    pl = pipeline
    pl.fit(x, y)

    train_sizes, train_scores, test_scores = learning_curve(pl, x, y, n_jobs=-1, cv=cv,
                                                            train_sizes=np.linspace(.1, 1.0, 5), verbose=0)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title(title)
    plt.legend(loc="best")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.gca().invert_yaxis()

    # box-like grid
    plt.grid()

    # plot the std deviation as a transparent range at each training set size
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                     alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")

    # plot the average training and test score lines at each training set size
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    # sizes the window for readability and displays the plot
    # shows error from 0 to 1.1
    plt.ylim(-.1, 1.1)
    plt.show()


if __name__ == '__main__':
    cve = CountVectorizerExtractor()
    tve = TfIdfVectorizerExtractor(cve)
    nb_classifier = NaiveBayesClassifier(cve=cve, tfidfe=tve)
    rf_classifier = RandomForestClassification(cve=cve, tfidfe=tve)
    lr_classifier = LogisticRegressionClassifier(cve=cve, tfidfe=tve)
    pickle.dump(rf_classifier.rf_pipeline_ngram_cv_normalized, open('model.sav', 'wb'))
    # plot_learing_curve(nb_classifier.nb_pipeline_cv, 'NB Count Vector',
    #                    cve.df['Headline'] + cve.df['articleBody'], cve.df['Stance'])
    # plot_learing_curve(nb_classifier.nb_pipeline_cv_normalized, 'NB Count Vector Normalised',
    #                    cve.df_normalized['Headline'] + cve.df_normalized['articleBody'], cve.df['Stance'])
    # plot_learing_curve(nb_classifier.nb_pipeline_ngram_cv, 'NB Count Vector N-Grams',
    #                    cve.df_normalized['Headline'] + cve.df_normalized['articleBody'], cve.df['Stance'])
    # plot_learing_curve(nb_classifier.nb_pipeline_ngram_cv_normalized, 'NB Count Vector N-Grams Normalised',
    #                    cve.df_normalized['Headline'] + cve.df_normalized['articleBody'], cve.df['Stance'])
    # plot_learing_curve(nb_classifier.nb_pipeline_pos_tag_cv, 'NB Count Vector POS Taggiing',
    #                    CsvReader.train_pos_tag['Headline'] + CsvReader.train_pos_tag['articleBody'], cve.df['Stance'])
    # plot_learing_curve(lr_classifier.lr_pipeline_cv, 'LR Count Vector',
    #                    cve.df['Headline'] + cve.df['articleBody'], cve.df['Stance'])
    # plot_learing_curve(lr_classifier.lr_pipeline_cv_normalized, 'LR Count Vector Normalised',
    #                    cve.df_normalized['Headline'] + cve.df_normalized['articleBody'], cve.df['Stance'])
    # plot_learing_curve(lr_classifier.lr_pipeline_ngram_cv, 'LR Count Vector N-Grams',
    #                    cve.df_normalized['Headline'] + cve.df_normalized['articleBody'], cve.df['Stance'])
    # plot_learing_curve(lr_classifier.lr_pipeline_ngram_cv_normalized, 'LR Count Vector N-Grams Normalised',
    #                    cve.df_normalized['Headline'] + cve.df_normalized['articleBody'], cve.df['Stance'])
    # plot_learing_curve(lr_classifier.lr_pipeline_pos_tag_cv, 'LR Count Vector POS Taggiing',
    #                    CsvReader.train_pos_tag['Headline'] + CsvReader.train_pos_tag['articleBody'], cve.df['Stance'])
    # plot_learing_curve(rf_classifier.rf_pipeline_cv, 'RF Count Vector',
    #                    cve.df['Headline'] + cve.df['articleBody'], cve.df['Stance'])
    # plot_learing_curve(rf_classifier.rf_pipeline_cv_normalized, 'RF Count Vector Normalised',
    #                    cve.df_normalized['Headline'] + cve.df_normalized['articleBody'], cve.df['Stance'])
    # plot_learing_curve(rf_classifier.rf_pipeline_ngram_cv, 'RF Count Vector N-Grams',
    #                    cve.df_normalized['Headline'] + cve.df_normalized['articleBody'], cve.df['Stance'])
    # plot_learing_curve(rf_classifier.rf_pipeline_ngram_cv_normalized, 'RF Count Vector N-Grams Normalised',
    #                    cve.df_normalized['Headline'] + cve.df_normalized['articleBody'], cve.df['Stance'])
    # plot_learing_curve(rf_classifier.rf_pipeline_pos_tag_cv, 'RF Count Vector POS Taggiing',
    #                    CsvReader.train_pos_tag['Headline'] + CsvReader.train_pos_tag['articleBody'],
    #                    cve.df['Stance'])

