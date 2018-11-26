import pickle

import numpy
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

import DataFormatter

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
model = pickle.load(open('model.sav', 'rb'))


# cve = CountVectorizerExtractor()
# tve = TfIdfVectorizerExtractor(cve)
#
# rf_classifier = RandomForestClassification(cve=cve, tfidfe=tve)


# https://bizstandardnews.com/2018/11/26/robertson-god-will-reward-trump-america-for-returning-to-christian-values/#

@app.route('/')
@cross_origin()
def hello_world():
    url = request.args.get('url')
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    # content = DataFormatter.strip_html_tags(soup)

    headline = soup.find("h1", {"class": "entry-title"})

    body = ''
    for paragraph in soup.find("div", {"class": "entry-content"}).findAll("p"):
        body += DataFormatter.strip_html_tags(paragraph)

    content = {'headline': DataFormatter.strip_html_tags(headline),
               'body': body}
    # content_df = pandas.DataFrame(columns=['Body ID', 'Headline', 'articleBody'])
    # content_df.append(['0', DataFormatter.normalize_string(content['headline'])],
    #                   DataFormatter.normalize_string(content['body']))
    # content_df.loc[1]['Body ID'] =
    # content_df.loc[1]['Headline'] =
    # content_df.loc[1]['articleBody'] =
    # print(content_df)
    X_prediction = numpy.array([DataFormatter.normalize_string(content['headline'] + content['body'])])

    # print(X_prediction.reshape(-1,1))
    # print(X_prediction.reshape(-1,1).shape)

    prediction = model.predict(X_prediction)
    prob = model.predict_proba(X_prediction)
    # train_set = read_dataframe_train()
    # test_dataframe(train_set)
    # data_frame_to_csv(train_set)
    # create_distribution(train_set)
    return jsonify(prediction, prob)


if __name__ == '__main__':
    app.run()
