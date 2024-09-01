import pandas as pd
import streamlit as st
import joblib
from helper import *

count_vec = joblib.load('count_vectorizer.pkl')
mnb_model = joblib.load("disaster_or_not.pkl")


# Preprocessing
def text_preprocessing(text):
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
    text = pd.Series([text])

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
    return X

def predict(text):
    text_cleaned = text_preprocessing(text)
    # Count vectorizer
    input_tokens = count_vec.transform(text_cleaned)

    # MultiNB, Stacking, LogReg models predictions
    mnb_predict = mnb_model.predict(input_tokens)
    return mnb_predict[0]  # Return the first element from the prediction array


def main():
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Disaster Call Classification App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    input_text = st.text_input('Enter Disaster Call', "")

    with st.container():
        with st.columns(5)[2]:
            flag = st.button('Submit')

    with st.container():
        if flag:
            pred = predict(input_text)
            if pred == 1:
                result = "<div style='background-color:red;color:white;padding:20px;text-align:center;'>DISASTER CALL</div>"
            else:
                result = "<div style='background-color:green;color:white;padding:20px;text-align:center;'>DON'T PANIC. It's NOT a Disaster</div>"

            st.markdown(result, unsafe_allow_html=True)

    # if st.button("Predict"):
    #      pred = predict(input_text)
    #      if pred == 1:
    #          result = "It's a Disaster"
    #      else:
    #         result = "It's NOT a Disaster"
    # st.success(result)
    if st.button("Who built"):
        st.text("built by ofa using streamlit")



if __name__=='__main__':
    main()
