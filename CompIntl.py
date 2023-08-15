import pandas as pd
import spacy
from gensim.utils import simple_preprocess
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
import requests
from streamlit_lottie import st_lottie
import nltk
from nltk.corpus import stopwords
import streamlit as st;

def perform_topic_modeling(data_path, num_topics=5, passes=5):
    def preprocess_text(texts, allowed_postags=["NOUN", "VERB", "ADJ", "ADV"], stopwords=None):
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        texts_out = []

        if stopwords is None:
            stopwords = []

        for text in texts:
            doc = nlp(text)
            new_text = []
            for token in doc:
                if token.pos_ in allowed_postags:
                    if token.lemma_ != '-PRON-':
                        new_text.append(token.lemma_)
                    else:
                        new_text.append(token.text)
            final = " ".join(new_text)
            texts_out.append(final)
        return texts_out

    def clean_text(texts, stopwords):
        cleaned_texts = []
        for text in texts:
            new_text = simple_preprocess(text, deacc=True)
            new_text = [word for word in new_text if word not in stopwords and len(word) > 4]
            cleaned_texts.append(new_text)
        return cleaned_texts

    df = pd.read_csv(data_path)

    df_pos = df[df['reviews.doRecommend'] == True]
    df_neg = df[df['reviews.doRecommend'] == False]

    pos_text = df_pos['reviews.text']
    neg_text = df_neg['reviews.text']

    custom_sw = ['however', 'first']  # Your list of custom stopwords
    stopwords_list = stopwords.words("english")
    stopwords_list.extend(custom_sw)

    lemmatized_texts_pos = preprocess_text(pos_text)
    lemmatized_texts_neg = preprocess_text(neg_text)

    data_clean_pos = clean_text(lemmatized_texts_pos, stopwords_list)
    data_clean_neg = clean_text(lemmatized_texts_neg, stopwords_list)

    bigram = Phrases(data_clean_pos, min_count=5, threshold=10)
    trigram = Phrases(bigram[data_clean_pos], min_count=5, threshold=10)

    bow_pos = [trigram[sentence] for sentence in data_clean_pos]
    bow_neg = [trigram[sentence] for sentence in data_clean_neg]

    id2word_pos = Dictionary(bow_pos)
    corpus_matrix_pos = [id2word_pos.doc2bow(sent) for sent in bow_pos]

    id2word_neg = Dictionary(bow_neg)
    corpus_matrix_neg = [id2word_neg.doc2bow(sent) for sent in bow_neg]

    lda_model_pos = LdaModel(
        corpus=corpus_matrix_pos,
        id2word=id2word_pos,
        num_topics=num_topics,
        random_state=100,
        update_every=1,
        chunksize=100,
        passes=passes,
        alpha='auto',
        per_word_topics=True
    )

    topics_pos = lda_model_pos.show_topics(num_topics=num_topics, num_words=10, formatted=False)
    topic_pos_list = []  # List to store words for each topic

    for topic_id, topic in topics_pos:
        topic_words = [word for word, _ in topic]
        topic_pos_list.append(topic_words) 


    lda_model_neg = LdaModel(
        corpus=corpus_matrix_neg,
        id2word=id2word_neg,
        num_topics=num_topics,
        random_state=100,
        update_every=1,
        chunksize=100,
        passes=passes,
        alpha='auto',
        per_word_topics=True
    )

    topics_neg = lda_model_neg.show_topics(num_topics=num_topics, num_words=10, formatted=False)
    topic_neg_list = []  # List to store words for each topic

    for topic_id, topic in topics_neg:
        topic_words = [word for word, _ in topic]
        topic_neg_list.append(topic_words)
    
    
    return topic_pos_list, topic_neg_list



# # Example usage
# data_path = "Code/Walmart Reviews.csv"  # Update the path to your CSV file
# perform_topic_modeling(data_path, num_topics=5, passes=5)
def load_lottieurl(url):
    r=requests.get(url)
    if r.status_code!=200:
        return None
    return r.json()

lottie_coding=load_lottieurl("https://lottie.host/23dbc176-5338-47c8-98f3-61a5379cafa7/Vysd3RBFnc.json")

def main():
    data_path = "Walmart Reviews.csv"  # Update the path to your CSV file
    # Perform topic modeling
    topics_pos_text, topics_neg_text = perform_topic_modeling(data_path, num_topics=5, passes=5)

    # Create columns for positive and negative reviews topics
    col1, col2 = st.columns(2)
    
    with col1:    
        st.title("Competitive Intelligence")
        st.subheader("Hi, This is an Advanced Competitive Intelligence System")
        st.write("Enter the product link:")
        product_link = st.text_input("Product Link", "")
        if st.button("Analyze"):
            st.write(f"Analyzing the product : {product_link}")

            l_col,r_col=st.columns(2)
            with l_col:  
                st.write("Best Perfoming Categories:")            
                for words in topics_pos_text:
                    st.write(f" {words}")
            with r_col:
                st.write("Areas for Improvement:")
                for words in topics_neg_text:
                    st.write(f" {words}")
    with col2:
        st_lottie(lottie_coding, height=300, key='coding')

if __name__ == "__main__":
    main()







