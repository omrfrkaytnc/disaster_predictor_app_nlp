def remove_html_tags(text):
    import re
    clean_text = re.sub('<.*?>', '', text)
    return clean_text

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    clean_text = re.sub(url_pattern, '', text)
    return clean_text

def remove_punctuation(text):
    punctuation = string.punctuation
    clean_text = text.translate(str.maketrans('', '', punctuation))
    return clean_text

def replace_emojis_with_meanings(text):
    def replace(match):
        emoji_char = match.group()
        emoji_meaning = emoji.demojize(emoji_char)
        return emoji_meaning

    emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"
                            u"\U0001F300-\U0001F5FF"
                            u"\U0001F680-\U0001F6FF"
                            u"\U0001F1E0-\U0001F1FF"
                            u"\U00002500-\U00002BEF"
                            u"\U00002702-\U000027B0"
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            u"\U0001f926-\U0001f937"
                            u"\U00010000-\U0010ffff"
                            u"\u2640-\u2642"
                            u"\u2600-\u2B55"
                            u"\u200d"
                            u"\u23cf"
                            u"\u23e9"
                            u"\u231a"
                            u"\ufe0f"
                            u"\u3030"
                            "]+", flags=re.UNICODE)
    text_with_meanings = emoji_pattern.sub(replace, text)
    return text_with_meanings

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)




import re
import string

# Matplotlib
import joblib
# NLTK
import nltk
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
# Scikit-learn
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

# TextBlob
# WordCloud

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def clear_text(text):
    """
    Cleans the input text by removing URLs, special characters, emojis, HTML tags,
    punctuation, and numbers.

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)

    # Remove emojis
    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
    text = emoji_pattern.sub('', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d', '', text)

    return text


chat_words_mapping = {
    "lol": "laughing out loud",
    "brb": "be right back",
    "btw": "by the way",
    "afk": "away from keyboard",
    "rofl": "rolling on the floor laughing",
    "ttyl": "talk to you later",
    "np": "no problem",
    "thx": "thanks",
    "omg": "oh my god",
    "idk": "I don't know",
    "np": "no problem",
    "gg": "good game",
    "g2g": "got to go",
    "b4": "before",
    "cu": "see you",
    "yw": "you're welcome",
    "wtf": "what the f*ck",
    "imho": "in my humble opinion",
    "jk": "just kidding",
    "gf": "girlfriend",
    "bf": "boyfriend",
    "u": "you",
    "r": "are",
    "2": "to",
    "4": "for",
    "b": "be",
    "c": "see",
    "y": "why",
    "tho": "though",
    "smh": "shaking my head",
    "lolz": "laughing out loud",
    "h8": "hate",
    "luv": "love",
    "pls": "please",
    "sry": "sorry",
    "tbh": "to be honest",
    "omw": "on my way",
    "omw2syg": "on my way to see your girlfriend",
    "atb": "all the best",
    "aka": "also known as",
    "adih": "another day in hell",
    "aymm": "are you my mother?",
    "ruok": "are you ok?",
    "aamof": "as a matter of fact",
    "afaict": "as far as i can tell",
    "afaik": "as far as i know",
    "afair": "as far as i remember",
    "afaic": "as far as i’m concerned",
    "asap": "as soon as possible",
    "ama": "ask me anything",
    "atm": "at the moment",
    "ayor": "at your own risk",
    "afk": "away from keyboard",
    "b@u": "back at you",
    "bbias": "be back in a sec",
    "brb": "be right back",
    "bc": "because",
    "b4": "before",
    "bae": "before anyone else",
    "bff": "best friends forever",
    "bsaaw": "big smile and a wink",
    "bf": "boyfriend",
    "bump": "bring up my post",
    "bro": "brother",
    "bwl": "bursting with laughter",
    "btw": "by the way",
    "bbbg": "bye bye be good",
    "csl": "can't stop laughing",
    "cip": "commercially important person",
    "cwot": "complete waste of time",
    "gratz": "congratulations",
    "qq": "crying",
    "d8": "date",
    "dm": "direct message",
    "diy": "do it yourself",
    "dbmib": "don't bother me i'm busy",
    "dwh": "during work hours",
    "emb": "early morning business meeting",
    "e123": "easy as one, two, three",
    "f2f": "face to face",
    "fomo": "fear of missing out",
    "4ao": "for adults only",
    "fawc": "for anyone who cares",
    "ftl": "for the loss",
    "fyi": "for your information",
    "4ever": "forever",
    "fimh": "forever in my heart",
    "fka": "formerly known as",
    "faq": "frequently asked questions",
    "gahoy": "get a hold of yourself",
    "goi": "get over it",
    "gf": "girlfriend",
    "gfn": "gone for now",
    "gg": "good game",
    "gl": "good luck",
    "gr8": "great",
    "gmta": "great minds think alike",
    "goat": "greatest of all time",
    "hb2u": "happy birthday to you",
    "hf": "have fun",
    "xoxo": "hugs and kisses",
    "idc": "i don't care",
    "idk": "i don't know",
    "ifyp": "i feel your pain",
    "ik": "i know",
    "ily/ilu": "i love you",
    "ilysm/lysm": "i love you so much",
    "imu": "i miss you",
    "iirc": "if i remember correctly",
    "icymi": "in case you missed it",
    "imo": "in my opinion",
    "irl": "in real life",
    "j4f": "just for fun",
    "jic": "just in case",
    "jk": "just kidding",
    "jsyk": "just so you know",
    "l8": "late",
    "l8r": "later",
    "lol": "laughing out loud",
    "lmk": "let me know",
    "mfw": "my face when",
    "nvm": "nevermind",
    "nmy": "nice meeting you",
    "np": "no problem",
    "nagi": "not a good idea",
    "n/a": "not available",
    "nbd": "not big deal",
    "nfs": "not for sale",
    "nm": "not much",
    "nsfl": "not safe for life",
    "nsfw": "not safe for work",
    "omg": "oh my god",
    "omw": "on my way",
    "oc": "original content",
    "omdb": "over my dead body",
    "oh": "overheard",
    "ppl": "people",
    "potd": "photo of the day",
    "pls": "please",
    "ptb": "please text back",
    "pov": "point of view",
    "ps": "post script",
    "rbtl": "read between the lines",
    "rsvp": "respondez s’il vous plaît (french)",
    "rofl": "rolling on the floor laughing",
    "sfw": "safe for work",
    "ssdd": "same stuff, different day",
    "c u": "see you",
    "cyt": "see you tomorrow",
    "srsly": "seriously",
    "smh": "shaking my head",
    "sis": "sister",
    "zzz": "sleep",
    "soml": "story of my life",
    "ttyl": "talk to you later",
    "time": "tears in my eyes",
    "tgif": "thank god, it’s friday",
    "thx": "thanks",
    "tia": "thanks in advance",
    "tbt": "throwback thursday",
    "tbc": "to be continued",
    "tbh": "to be honest",
    "til": "today i learned",
    "2nite": "tonight",
    "tl;dr": "too long; didn’t read",
    "tmi": "too much information",
    "tntl": "trying not to laugh",
    "vip": "very important person",
    "w8": "wait",
    "wyd": "what are you doing?",
    "sup?": "what’s up?",
    "wywh": "wish you were here",
    "wfm": "works for me",
    "u": "you",
    "ygtr": "you got that right",
    "ynk": "you never know",
    "hbd": "happy birthday",
    "smh": "shaking my head",
    "idk": "I don't know",
    "imho": "in my humble opinion",
    "tbh": "to be honest",
    "omg": "oh my god",
    "yolo": "you only live once",
    "fml": "fuck my life",
    "tl;dr": "too long; didn't read",
    "fyi": "for your information",
    "ttyl": "talk to you later",
    "bff": "best friends forever",
    "bday": "birthday",
    "gr8": "great",
    "omw": "on my way",
    "lmk": "let me know",
    "g2g": "got to go",
    "asap": "as soon as possible",
    "ttys": "talk to you soon",
    "gfy": "good for you",
    "tl;dr": "too long; didn't read",
    "bbl": "be back later",
    "fyi": "for your information",
    "plz": "please",
    "np": "no problem",
    "hmu": "hit me up",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "icymi": "in case you missed it",
}


def expand_chat_words(text):
    """
    Expands common chat abbreviations into their full forms based on a predefined dictionary.

    Args:
        text (str): The text containing abbreviations.
        chat_words_mapping (dict): A dictionary mapping abbreviations to their full forms.

    Returns:
        str: The text with abbreviations replaced by their full forms.
    """
    words = text.split()
    expanded_words = [chat_words_mapping.get(word.lower(), word) for word in words]
    return ' '.join(expanded_words)


def get_wordnet_pos(tag):
    """
    Converts NLTK part-of-speech tags to WordNet format.

    Args:
        tag (str): The part-of-speech tag from NLTK.

    Returns:
        str: The corresponding WordNet part-of-speech tag.
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def df_preprocessing(dataframe):
    """
    Preprocesses the dataframe by cleaning the text, expanding abbreviations, tokenizing,
    converting to lowercase, removing stopwords, tagging parts of speech, lemmatizing,
    and preparing the text for further processing.

    Args:
        dataframe (pd.DataFrame): The input dataframe containing 'text' and 'target' columns.

    Returns:
        tuple: A tuple containing:
            - X (list): The preprocessed text.
            - y (pd.Series): The target values.
    """
    df = dataframe[['text', 'target']]
    text = df['text']
    text = text.apply(clear_text)
    text = text.apply(lambda x: expand_chat_words(x))

    # Tokenizing the text
    text = text.apply(word_tokenize)

    # Lowercasing the tokens
    text = text.apply(lambda x: [word.lower() for word in x])

    # Removing stopwords
    stop = set(stopwords.words('english'))
    text = text.apply(lambda x: [word for word in x if word not in stop])

    # Applying part of speech tags
    text = text.apply(nltk.tag.pos_tag)

    # Converting part of speech tags to WordNet format
    text = text.apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])

    # Lemmatizing the words
    wnl = WordNetLemmatizer()
    text = text.apply(lambda x: [wnl.lemmatize(word, tag) for word, tag in x])
    text = text.apply(lambda x: [word for word in x if word not in stop])
    text = [' '.join(map(str, l)) for l in text]

    X = text
    y = df['target']

    return X, y


def count_vectorizer(text):
    """
    Applies Count vectorization to the input text.

    Args:
        text (list of str): The list of text documents.

    Returns:
        sparse matrix: The Count vectorized representation of the text.
    """
    count_vec = CountVectorizer()
    text_vec = count_vec.fit_transform(text)
    X = text_vec
    return X, count_vec


def train_and_tune_model(X, y):
    nb_model = MultinomialNB()
    param_grid = {'alpha': [0.01, 0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
    grid_search = GridSearchCV(nb_model, param_grid, cv=5, n_jobs=-1, verbose=True)
    grid_search.fit(X, y)

    # Setting the best parameters
    nb_final = nb_model.set_params(**grid_search.best_params_)
    nb_final.fit(X, y)

    return nb_final


# for app.py
def text_preprocessing(text):
    # Convert text to a pandas Series
    text_series = pd.Series([text])

    # Apply preprocessing functions
    text_series = text_series.apply(clear_text)
    text_series = text_series.apply(lambda x: expand_chat_words(x))

    # Tokenizing the tweet base texts.
    text_series = text_series.apply(word_tokenize)

    # Lower casing clean text.
    text_series = text_series.apply(lambda x: [word.lower() for word in x])

    # Removing stopwords.
    stop = set(stopwords.words('english'))
    text_series = text_series.apply(lambda x: [word for word in x if word not in stop])

    # Applying part of speech tags.
    text_series = text_series.apply(nltk.tag.pos_tag)

    # Converting part of speeches to wordnet format.
    text_series = text_series.apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])

    # Applying word lemmatizer.
    wnl = WordNetLemmatizer()
    text_series = text_series.apply(lambda x: [wnl.lemmatize(word, tag) for word, tag in x])
    text_series = text_series.apply(lambda x: [word for word in x if word not in stop])

    # Joining words into a single string
    text_processed = [' '.join(map(str, l)) for l in text_series]

    return text_processed







