"""
File: train.py
------------------------------------
Topic Modelling Pipeline --- Stages of 
1. Text Data Preprocessing (Filtering of No Comment or Not Applicable comments from csv file)
2. Model Building
"""

#Import statements
import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)
from umap import UMAP #UMAP Dimensionality Reduction
from bertopic import BERTopic #Topic Modelling
from sklearn.feature_extraction.text import CountVectorizer #CountVectorizer Bag of Words
from sentence_transformers import SentenceTransformer #Embedding model

#import modules
import data_preprocessing #data preprocessing module

#Global Constants
DATA_PATH = r"..\..\masds002\DS2-assessment-Simulated-Employee-Feedback.xlsx" #Path to raw data

def main():
    data = pd.read_excel(DATA_PATH)
    print(len(data))
    data = data_preprocessing.remove_na_comments(data)
    print(len(data))

    docs = data["employee_feedback"]
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(docs)

    vectorizer_model = CountVectorizer(stop_words="english")
    #sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    sentence_model = SentenceTransformer("all-mpnet-base-v2")
    #embeddings = sentence_model.encode(docs, show_progress_bar=False)
    umap_model = UMAP(n_neighbors=3, n_components=5, min_dist=0.0, metric='cosine',random_state=42)
    topic_model = BERTopic(vectorizer_model=vectorizer_model,min_topic_size=5, 
                        umap_model=umap_model,embedding_model=sentence_model)
    topics, probs = topic_model.fit_transform(docs)

    topic_model.get_topic_info().to_csv("results.csv")

    topic_model.visualize_heatmap()



if __name__ == "__main__":
    main()
    


