import os
import xml.etree.ElementTree as ET
import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import hstack
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

def load_data(data_directory, filename):
    source = os.path.join(data_directory, filename)
    tree = ET.parse(source)
    root = tree.getroot()
    texts_list = []
    opinions_list = []

    for review in root.findall('Review'):
        text_string = ""
        opinion_inner_list = []

        for sentence in review.findall('sentences/sentence'):
            text_string = text_string + " " + sentence.find('text').text
        texts_list.append(text_string)

        for opinion in review.findall('Opinions/Opinion'):
            opinions_dict = {
                opinion.get('category').replace('#', '_'):opinion.get('polarity')
            }
            opinion_inner_list.append(opinions_dict)
        opinions_list.append(opinion_inner_list)

    return texts_list, opinions_list


def tokenize_user_data(user_input):
    tokens_list  = word_tokenize(user_input.lower())
    return tokens_list


# tokenize sentences into tokens and then save under list format.
def tokenize_data(texts_lists):
    tokens_list = []

    for sentences in texts_lists:
        token_list = word_tokenize(sentences.lower())   # lowercase and then tokenize the sentence
        tokens_list.append(token_list)
    return tokens_list


# trimming data and tokens lemmatization
def clean_data(tokens_list):
    cleaned_tokens_list = []
    lammatizer = nltk.stem.WordNetLemmatizer()

    for tokens in tokens_list:
        # remove punctuation from each token
        table = str.maketrans('', '', punctuation)
        tokens = [word.translate(table) for word in tokens]
        # remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if not word in stop_words]
        # remove remaining tokens that are not alphabetic
        tokens = [word for word in tokens if word.isalpha()]
        # lemmatization
        tokens = [lammatizer.lemmatize(word, 'v') for word in tokens]
        cleaned_tokens_list.append(tokens)

    texts_list = []

    # append tokens in the same review to a text
    for tokens in cleaned_tokens_list:
        text = []
        for token in tokens:
            text.append(token)
        texts_list.append(' '.join(text))

    return texts_list


# select the most common aspects to be labels for Aspect Detection
def get_most_common_aspects(opinions_list):
    opinion = []

    for inner_list in opinions_list:
        for _dict in inner_list:
            for key in _dict:
                opinion.append(key)
    most_common_aspects = [k for k,v in nltk.FreqDist(opinion).most_common(20)] # just get 20 aspects which is the most popular aspect

    return most_common_aspects


# construct a data frame from data after preprocessed
def get_data_frame(texts_list, opinions_list, most_common_aspects):
    data = {'Review': texts_list} # Review column contains list of tokens
    df = pd.DataFrame(data)

    if opinions_list:
        for inner_list in opinions_list:
            for _dict in inner_list:
                for key in _dict:
                    if key in most_common_aspects:
                        df.loc[opinions_list.index(inner_list), key] = _dict[key]

    return df


# construct data frames for #2
# just reach positive aspects
def get_pos_data_frame(df, most_common_aspects):

    for common_aspect in most_common_aspects:
        df[common_aspect] = df[common_aspect].replace(['positive'], [1])
        df[common_aspect] = df[common_aspect].replace(['negative', 'neutral', 'conflict'], [0, 0, 0])
    df = df.fillna(0)

    return df


# just reach negative aspects
def get_neg_data_frame(df, most_common_aspects):

    for common_aspect in most_common_aspects:
        df[common_aspect] = df[common_aspect].replace(['negative'], [1])
        df[common_aspect] = df[common_aspect].replace(['positive', 'neutral', 'conflict'], [0, 0, 0])
    df = df.fillna(0)

    return df


# just reach neutral and conflict aspects
def get_neu_data_frame(df, most_common_aspects):

    for common_aspect in most_common_aspects:
        df[common_aspect] = df[common_aspect].replace(['neutral', 'conflict'], [1, 1])
        df[common_aspect] = df[common_aspect].replace(['negative', 'positive'], [0, 0])
    df = df.fillna(0)

    return df


# normalize the aspect data frame for #1 (aspects mentioned are assigned to 1)
def normalize_aspect_data_frame(df, most_common_aspects):

    for aspect in most_common_aspects:
        df[aspect] = df[aspect].replace(['positive', 'negative', 'neutral', 'conflict'], [1, 1, 1, 1])
    df = df.fillna(0)

    return df


def create_dict_of_aspect(y_train, most_common_aspects):
    positions = [] # contains indexes of aspects implied in review

    for aspects_vector in y_train:
        positions.append([i for i, j in enumerate(aspects_vector) if j == 1]) # i <- index of position; j <- value at this position

    sorted_common_aspects = sorted(most_common_aspects)
    dict_of_aspects = []

    for position in positions:
        inner_dict = {}

        for aspect in sorted_common_aspects:
            if sorted_common_aspects.index(aspect) in position:
                inner_dict[aspect] = 5
            else:
                inner_dict[aspect] = 0
        dict_of_aspects.append(inner_dict)

    return dict_of_aspects


# classify positive, negative, or neutral aspects
def classify_sentiment(df_train, X_train_aspect_dtm, df_test, X_test_aspect_dtm):
    df_train = df_train.reindex_axis(sorted(df_train.columns), axis=1)
    X_train = df_train.Review
    y_train = df_train.drop('Review', 1)
    y_train = np.asarray(y_train, dtype=np.int64)

    df_test = df_test.reindex_axis(sorted(df_test.columns), axis=1)
    X_test = df_test.Review
    y_test = df_test.drop('Review', 1)
    y_test = np.asarray(y_test, dtype=np.int64)

    sentiment_vector = CountVectorizer(stop_words='english', ngram_range=(1, 2))
    X_train_dtm = sentiment_vector.fit_transform(X_train)
    X_test_dtm = sentiment_vector.transform(X_test)

    # combining original and extra train data
    X_train_dtm = hstack((X_train_dtm, X_train_aspect_dtm))
    X_test_dtm = hstack((X_test_dtm, X_test_aspect_dtm))

    nb_classifier = OneVsRestClassifier(MultinomialNB()).fit(X_train_dtm, y_train)
    y_pred_class_nb = nb_classifier.predict(X_test_dtm)

    print(metrics.accuracy_score(y_test, y_pred_class_nb))
    print(metrics.classification_report(y_test, y_pred_class_nb))

    rf_classifier = OneVsRestClassifier(RandomForestClassifier()).fit(X_train_dtm, y_train)
    y_pred_class_rf = rf_classifier.predict(X_test_dtm)

    print(metrics.accuracy_score(y_test, y_pred_class_rf))
    print(metrics.classification_report(y_test, y_pred_class_rf))

    svm_classifier = OneVsRestClassifier(svm.SVC()).fit(X_train_dtm, y_train)
    y_pred_class_svm = svm_classifier.predict(X_test_dtm)
    print(metrics.accuracy_score(y_test, y_pred_class_svm))
    print(metrics.classification_report(y_test, y_pred_class_svm))

    return nb_classifier, rf_classifier, svm_classifier


def main():
    ROOT_PATH = "/home/vng/Documents/KD-DM/Project3/sentiment-analysis"
    data_directory = os.path.join(ROOT_PATH, "data")

    # load data for pre-processing
    train_texts_list, train_opinions_list = load_data(data_directory, "Training.xml")
    test_texts_list, test_opinions_list = load_data(data_directory, "Testing.xml")

    # pre-process training and testing data
    train_tokens_list = tokenize_data(train_texts_list)
    train_texts_list = clean_data(train_tokens_list)

    test_tokens_list = tokenize_data(test_texts_list)
    test_texts_list = clean_data(test_tokens_list)

    # get 20 most common aspects
    most_common_aspects = get_most_common_aspects(train_opinions_list)
    wr = open("./data/common-aspects.txt", "w")
    for aspect in most_common_aspects:
        wr.write(aspect + "\n")
    wr.close()

#1. Aspect Detection
    # convert data format to be corresponding with fit_transform() method
    df_train = get_data_frame(train_texts_list, train_opinions_list, most_common_aspects)
    df_train.to_csv("./data/training-features.csv")
    df = normalize_aspect_data_frame(df_train, most_common_aspects) # aspects mentioned are assigned to 1
    df_train_aspect = df.reindex_axis(sorted(df.columns), axis=1) # re-arrange columns in df
    X_train = df_train_aspect.Review # split Review column of df to X_train set
    y_train = df_train_aspect.drop('Review', 1) # the rest of df is y_train
    y_train = np.asarray(y_train, dtype=np.int64)

    df_test = get_data_frame(test_texts_list, test_opinions_list, most_common_aspects)
    df_test.to_csv("./data/testing-features.csv")
    df = normalize_aspect_data_frame(df_test, most_common_aspects)
    df_test_aspect = df.reindex_axis(sorted(df.columns), axis=1)
    X_test = df_test_aspect.Review
    y_test = df_test_aspect.drop('Review', 1)
    y_test = np.asarray(y_test, dtype=np.int64)

    vect = CountVectorizer(lowercase=False, max_df=1.0, stop_words='english', max_features=2000)
    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)

    nb_classifier = OneVsRestClassifier(MultinomialNB()).fit(X_train_dtm, y_train)
    y_predicted_nb = nb_classifier.predict(X_test_dtm)
#    print(metrics.accuracy_score(y_test, y_predicted_nb))
#    print(metrics.classification_report(y_test, y_predicted_nb))
#    wr = open("./data/predicted-aspects.txt", "w")
#    for aspect in y_predicted_nb:
#        wr.write(str(aspect) + "\n")
#    wr.close()

    rf_classifier = OneVsRestClassifier(RandomForestClassifier()).fit(X_train_dtm, y_train)
    y_predicted_rf = rf_classifier.predict(X_test_dtm)

#    print(metrics.accuracy_score(y_test, y_predicted_rf))
#    print(metrics.classification_report(y_test, y_predicted_rf))

    svm_classifier = OneVsRestClassifier(svm.SVC(C=1.0, kernel='linear')).fit(X_train_dtm, y_train)
    y_predicted_svm = svm_classifier.predict(X_test_dtm)
#    print(metrics.classification_report(y_test, y_predicted_svm))
#2. Sentiment Detection for each Aspect
    # construct extra data
    dict_of_aspects = create_dict_of_aspect(y_train, most_common_aspects)
    aspect_vectorizer = DictVectorizer()
    X_train_aspect_dtm = aspect_vectorizer.fit_transform(dict_of_aspects)

    dict_of_aspects = create_dict_of_aspect(y_test, most_common_aspects)
    X_test_aspect_dtm = aspect_vectorizer.transform(dict_of_aspects)

    # construct original train data
    df_train = get_data_frame(train_texts_list, train_opinions_list, most_common_aspects)
    df_test = get_data_frame(test_texts_list, test_opinions_list, most_common_aspects)
    # for positive aspect detection
    df_train_pos = get_pos_data_frame(df_train, most_common_aspects)
    df_test_pos = get_pos_data_frame(df_test, most_common_aspects)
    # for negative aspect detection
    df_train = get_data_frame(train_texts_list, train_opinions_list, most_common_aspects)
    df_test = get_data_frame(test_texts_list, test_opinions_list, most_common_aspects)
    df_train_neg = get_neg_data_frame(df_train, most_common_aspects)
    df_test_neg = get_neg_data_frame(df_test, most_common_aspects)
    # for neutral or conflict aspect detection
    df_train = get_data_frame(train_texts_list, train_opinions_list, most_common_aspects)
    df_test = get_data_frame(test_texts_list, test_opinions_list, most_common_aspects)
    df_train_neu = get_neu_data_frame(df_train, most_common_aspects)
    df_test_neu = get_neu_data_frame(df_test, most_common_aspects)

    nb_pos_aspect_classifier, rf_pos_aspect_classifier, svm_pos_aspect_classifier = classify_sentiment(df_train_pos, X_train_aspect_dtm, df_test_pos, X_test_aspect_dtm)
    nb_neg_aspect_classifier, rf_neg_aspect_classifier, svm_neg_aspect_classifier = classify_sentiment(df_train_neg, X_train_aspect_dtm, df_test_neg, X_test_aspect_dtm)
    nb_neu_aspect_classifier, rf_neu_aspect_classifier, svm_neu_aspect_classifier = classify_sentiment(df_train_neu, X_train_aspect_dtm, df_test_neu, X_test_aspect_dtm)


# For an input from keyboard
#    user_input = input("Enter a laptop review: \n")
#    user_tokens = tokenize_user_data(user_input)
#    user_tokens_list = []
#    user_tokens_list.append(user_tokens)
#    user_texts_list = clean_data(user_tokens_list)
#    user_input = pd.Series(user_texts_list)
#    user_input_dtm = vect.transform(user_input)

#    predicted_aspects_nb = nb_classifier.predict(user_input_dtm)
#    predicted_aspects_rf = rf_classifier.predict(user_input_dtm)
#    predicted_aspects_svm = svm_classifier.predict(user_input_dtm)
#    print(str(predicted_aspects_nb))
#    print(str(predicted_aspects_rf))
#    print(str(predicted_aspects_svm))

#    extra_features = create_dict_of_aspect(predicted_aspects_svm, most_common_aspects)
#    extra_features_dtm = DictVectorizer().fit_transform(extra_features)

#    print(user_input_dtm)
#    print(extra_features_dtm)
#    user_input_aspect_dtm = hstack((user_input_dtm, extra_features_dtm))
#    print(user_input_aspect_dtm)
#    pos_aspect = svm_pos_aspect_classifier.predict(user_input_aspect_dtm)
#    print(pos_aspect)



if __name__ == '__main__':
    main()