import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix, classification_report, make_scorer  # noqa: E501
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")


def do_my_study():
    raw_df = get_data()

    pca_train, pca_test, y_train, y_test = preproc(raw_df)

    # Sanity check
    # print(df)

    # LR_modeling(pca_train, pca_test, y_train, y_test)
    RF_modeling(pca_train, pca_test, y_train, y_test)
    # find_score_and_best(df)


def get_data():
    df = pd.read_csv('reviews.csv')

    return df


def preproc(df):

    # Create new column "rating" based on 50 to 5.0, and so on
    df['rating'] = df['processed_rating']/10

    # Creating new column "churn" based on if "processed_rating" is > 35
    df['churn'] = df['rating'].apply(lambda x: 1 if x > 3.5 else 0)

    # Tokenize "processed_review"
    df['tokenized_review'] = df['processed_review'].apply(doc_preparer)

    # Pre-moldeling - train test split
    X = df['tokenized_review']
    y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=760397, stratify=y)  # noqa: E501

    # Vectorizing: Cutting off the top 10% and bottom 5% of words
    # in vectorized documents to keep the more meaningful words.
    tfidf_train = TfidfVectorizer(sublinear_tf=True,
                                  max_df=.9, min_df=.05,  ngram_range=(1, 1))
    train_features = tfidf_train.fit_transform(X_train).toarray()
    test_features = tfidf_train.transform(X_test).toarray()

    pca = PCA(n_components=0.9, random_state=1)
    pca_train = pca.fit_transform(train_features)
    pca_test = pca.transform(test_features)

    return pca_train, pca_test, y_train, y_test


def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def doc_preparer(doc):

    stop_words = stopwords.words('english')

    regex_token = RegexpTokenizer(r"([a-zA-Z]+(?:’[a-z]+)?)")
    doc = regex_token.tokenize(doc)
    doc = [word.lower() for word in doc]
    doc = [word for word in doc if word not in stop_words]
    doc = pos_tag(doc)
    doc = [(word[0], get_wordnet_pos(word[1])) for word in doc]
    lemmatizer = WordNetLemmatizer()
    doc = [lemmatizer.lemmatize(word[0], word[1]) for word in doc]
    return ' '.join(doc)


def metrics_score(train_preds, y_train, test_preds, y_test):
    print(f"Training Accuracy:\t{accuracy_score(y_train, train_preds):.4}",
          f"\tTesting Accuracy:\t{accuracy_score(y_test, test_preds):.4}")
    print(f"Training Precision:\t{precision_score(y_train, train_preds, average='weighted'):.4}",  # noqa: E501
          f"\tTesting Precision:\t{precision_score(y_test, test_preds, average='weighted'):.4}")  # noqa: E501
    print(f"Training Recall:\t{recall_score(y_train, train_preds, average='weighted'):.4}",  # noqa: E501
          f"\tTesting Recall:\t\t{recall_score(y_test, test_preds, average='weighted'):.4}")  # noqa: E501
    print(f"Training F1:\t\t{f1_score(y_train, train_preds, average='weighted'):.4}",  # noqa: E501
          f"\tTesting F1:\t\t{f1_score(y_test, test_preds, average='weighted'):.4}")  # noqa: E501


def LR_modeling(pca_train, pca_test, y_train, y_test):

    logistic_pipe = linear_model.LogisticRegression(class_weight='balanced',
                                                    random_state=760397)
    pipe = Pipeline(steps=[('STD_SCALER', MinMaxScaler()),
                           ('LR', logistic_pipe)])

    params = dict(LR__C=np.logspace(0, 1, 10),
                  LR__max_iter=[100],
                  LR__penalty=['none', 'l2'])

    LR_PCA_grid_search = GridSearchCV(estimator=pipe,
                                      scoring=make_scorer(f1_score,
                                                          average='weighted'),
                                      param_grid=params,
                                      cv=5, n_jobs=-1, verbose=2)

    LR_PCA_grid_search.fit(pca_train, y_train)
    LR_PCA_train_preds = LR_PCA_grid_search.best_estimator_.predict(pca_train)
    LR_PCA_test_preds = LR_PCA_grid_search.best_estimator_.predict(pca_test)

    print('\n')
    metrics_score(LR_PCA_train_preds, y_train, LR_PCA_test_preds, y_test)
    print("Best params: " + str(LR_PCA_grid_search.best_params_))
    print(classification_report(y_test, LR_PCA_test_preds))
    print(confusion_matrix(y_test, LR_PCA_test_preds).T)


def RF_modeling(pca_train, pca_test, y_train, y_test):
    rf_classifier = RandomForestClassifier(random_state=760397,
                                           class_weight='balanced')

    pipe = Pipeline(steps=[
        ('STD_SCALER', MinMaxScaler()),
        ('RF', rf_classifier)
    ])

    params = {
        'RF__n_estimators': [50, 100, 200],
        'RF__max_depth': [None, 10, 20],
        'RF__min_samples_split': [2, 5, 10],
        'RF__min_samples_leaf': [1, 2, 4]
    }

    RF_PCA_grid_search = GridSearchCV(
        estimator=pipe,
        scoring=make_scorer(f1_score, average='weighted'),
        param_grid=params,
        cv=5, n_jobs=-1, verbose=2
    )

    RF_PCA_grid_search.fit(pca_train, y_train)
    RF_PCA_train_preds = RF_PCA_grid_search.best_estimator_.predict(pca_train)
    RF_PCA_test_preds = RF_PCA_grid_search.best_estimator_.predict(pca_test)

    print('\n')
    metrics_score(RF_PCA_train_preds, y_train, RF_PCA_test_preds, y_test)
    print("Best params: " + str(RF_PCA_grid_search.best_params_))
    print(confusion_matrix(y_test, RF_PCA_test_preds).T)
    print(classification_report(y_test, RF_PCA_test_preds))


if __name__ == "__main__":

    do_my_study()
