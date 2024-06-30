# load required modules
from joblib import load
import pandas as pd
import fasttext
from googletrans import Translator
from nltk.stem import WordNetLemmatizer
from nltk.corpus import names
import nltk
from nltk.corpus import stopwords

# import spacy
# import pytextrank


# Download required nltk libraries (required once)
nltk.download("names")
nltk.download('wordnet')
nltk.download("stopwords")

# Initialize the lemmatizer and stopwords
all_names = set(names.words())
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')

# Load Spam model and corresponding vectorizer
spam_model = load(r"/model/lr_spam.pkl")
spam_vector = load(r"/model/vectorizer_spam.pkl")

# Load violence model and corresponding vectorizer
viol_model = load(r"/model/lr_violence.pkl")
viol_vector = load(r"/model/vectorizer_violence.pkl")

# Load positivity model and corresponding vectorizer
pos_model = load(r"/content/svm.pkl")
pos_vector = load(r"/content/vectorizer.pkl")

# load spacy model and pytextrank to spacy pipe
# nlp = spacy.load("en_core_web_sm")
# nlp.add_pipe("textrank")

# initialize google translator object
translator = Translator()

lang_model = fasttext.load_model(r'/model/lid.176.ftz')


def cleaned_text(docs):
    """
    Function to remove stopwords from text
    :param docs: string containing stopwords
    :return: string without stopwords
    """
    docs_cleaned = list()
    for doc in docs:
        doc = str(doc)
        doc = doc.lower()
        doc_cleaned = ' '.join(
            lemmatizer.lemmatize(word) for word in doc.split() if word not in all_names and word not in stop_words)
        docs_cleaned.append(doc_cleaned)
    return docs_cleaned


# def extract_keywords(text, n = 5):
#     """
#     Function to extract keywords from a sentence
#     :param text: String from which keywords are extracted
#     :param n: no. of keywords returned
#     :return: list of n most frequent keywords
#     """
#     return nlp(text)[:5]


def pred_pos(df: pd.DataFrame):
    """
    Predict results of positivity model
    :param df: DataFrame containing comments field
    :return: df with positivity field as result of prediction
    """
    cleaned_data = cleaned_text(df.comments)
    data_transformed = pos_vector.transform(cleaned_data)
    df["positivity"] = pos_model.predict(data_transformed)

    return df


def pred_spam(df: pd.DataFrame):
    """
    Predict results of Spam model
    :param df: DataFrame containing comments field
    :return: df with spam field as result of prediction
    """
    cleaned_data = cleaned_text(df.comments)
    data_transformed = spam_vector.transform(cleaned_data)
    df["spam"] = spam_model.predict(data_transformed)

    return df


def pred_viol(df: pd.DataFrame):
    """
    Predict results of violence model
    :param df: DataFrame containing comments field
    :return: df with violence field as result of prediction
    """
    cleaned_data = cleaned_text(df.comments)
    data_transformed = viol_vector.transform(cleaned_data)
    df["violence"] = viol_model.predict(data_transformed)

    return df


def func_lang(text):
    result = lang_model.predict(text, k=1)
    if result[1][0] > 0.8:
        return result[0][0][-2:]
    return "en"


def lang_detect(df: pd.DataFrame):
    """
    Detect language in comments
    :param df: DataFrame containing comments field
    :return: df with language field for corresponding comments
    """
    df["language"] = df.comments.apply(func_lang)

    return df


def translate_text(text, dest):
    text = text.rstrip()
    text = text.lstrip()
    try:
        print(text)
        return translator.translate(text, dest=dest).text
    except IndexError or TypeError:
        return "NA"


def translate_comments(df: pd.DataFrame, target="en"):
    """
    Translate comments into targeted language
    :param df: DataFrame containing comments field
    :param target: Language of translated comments
    :return: df with translated field for corresponding comments
    """
    df['translated_Text'] = df['comments'].apply(lambda x: translate_text(x, dest=target))
    return df


# def apply_keywords(df: pd.DataFrame):
#     """
#     Function for keyword extraction of text
#     :param df: DataFrame containing comments field
#     :return: df with keywords field corresponding to each comment
#     """
#     df["Keywords"] = df.comments.apply(extract_keywords)
#     return df


if __name__ == "__main__":
    df: pd.DataFrame = None
    df = pred_pos(df)
    df = pred_spam(df)
    df = pred_viol(df)
    df = lang_detect(df)
    df_en = translate_comments(df[df.language != "en"][:150])   # sending a subset of dataset due to api limits
