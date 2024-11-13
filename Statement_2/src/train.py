"""
File: train.py
------------------------------------
Topic Modelling Pipeline --- Stages of 
1. Text Data Preprocessing (Filtering of No Comment or Not Applicable comments from csv file)
2. Model Building
"""

#Import statements
import os #os file paths
import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)
from umap import UMAP #UMAP Dimensionality Reduction
from bertopic import BERTopic #Topic Modelling
from sklearn.feature_extraction.text import CountVectorizer #CountVectorizer Bag of Words
from sentence_transformers import SentenceTransformer #Embedding model
from ctransformers import AutoModelForCausalLM #HuggingFace LLM Auto-regressive task
from transformers import AutoTokenizer, pipeline #Tokenzier and high level pipeline
from bertopic.representation import TextGeneration #Text Generation task by Bertopic


#import modules
import data_preprocessing #data preprocessing module
import visualisation #visualisation module
import custom_prompts #custom prompts module

#Global Constants
DATA_PATH = r"..\..\masds002\DS2-assessment-Simulated-Employee-Feedback.xlsx" #Path to raw data
SAVE_VISUALISATION_PATH = r"..\visualisations" #Path to visualisation folder
TOPIC_RESULTS_PATH = r"..\results" #Path to results folder


def main():
    #Read the excel file
    data = pd.read_excel(DATA_PATH)
    #Perform Data Preprocessing
    data = data_preprocessing.remove_na_comments(data)

    #Retrieve only the relevant text column
    docs = data["employee_feedback"]

    #Create bertopic model and perform topic modelling on the docs
    topic_model = create_bertopic_model()
    topics, probs = topic_model.fit_transform(docs)

    #Save the results of topic modelling 
    topic_model.get_topic_info().to_csv(os.path.join(TOPIC_RESULTS_PATH,"topics_before_reduce.csv"))

    #Plot heatmap of topics after reducing the n neighbours
    visualisation.plot_heatmap(topic_model,"reduce_neighbour_heatmap.html")

    # Run the visualization with the topic-documents plot
    visualisation.plot_document(docs,topic_model,"reduce_neighbour_documents.html")

    # Run the visualization with the topic-documents plot
    visualisation.plot_data_document(docs,topic_model,"reduce_neighbour_data_documents.jpeg")

    # Reduce outliers using the `embeddings` strategy
    reduce_outlier_topics(topic_model,docs,topics)

    #  Run the visualization with the topic-documents plot after reducing outlier documents/rows
    visualisation.plot_document(docs,topic_model,"reduce_outlier_documents.html")

    # Run the visualization with the topic-documents plot after reducing outlier documents/rows
    visualisation.plot_data_document(docs,topic_model,"reduce_outliers_data_documents.jpeg")

    #Topics per the department
    topics_per_class = topic_model.topics_per_class(docs, classes=data.department)

    visualisation.plot_topics_dept(topics_per_class, topic_model, "department_topics.html")

    #Save the results of topic modelling 
    topic_model.get_topic_info().to_csv(os.path.join(TOPIC_RESULTS_PATH,"topics_after_reduce.csv"))



def reduce_outlier_topics(topic_model,docs,topics):
    """Attempts to classify outliers (-1 values) to a known topic provided that it has a minimum similarity that is 
    specified in the threshold parameter

    Args:
        topic_model: trained topic model with learnt topics
        docs: docs consisting of the text for the topic modelling
        topics: Respective topics for every document

    """    
    new_topics = topic_model.reduce_outliers(docs, topics, strategy="embeddings", threshold=0.3)
    vectorizer_model = CountVectorizer(stop_words="english")
    topic_model.update_topics(docs, topics=new_topics, vectorizer_model=vectorizer_model)


def create_bertopic_model():
    """Creates Bertopic Topic Model 

    Returns:
        topic_model: trained topic model with learnt topics

    """    
    representation_model = create_representation_model()
    vectorizer_model = CountVectorizer(stop_words="english")
    sentence_model = SentenceTransformer("all-mpnet-base-v2")
    umap_model = UMAP(n_neighbors=3, n_components=5, min_dist=0.0, metric='cosine',random_state=42)
    topic_model = BERTopic(vectorizer_model=vectorizer_model,min_topic_size=5, 
                           umap_model=umap_model,embedding_model=sentence_model,
                           representation_model=representation_model)
    return topic_model
    
def create_representation_model():
    """Creates Representation LLM Model for better profiling of the various topics to describe individuals

    Returns:
        representation_model: LLM model to have a better description of the text using customised prompts

    """    
    # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
    model = AutoModelForCausalLM.from_pretrained(
        "TheBloke/zephyr-7B-alpha-GGUF",
        model_file="zephyr-7b-alpha.Q4_K_M.gguf",
        model_type="mistral",
        gpu_layers=50,
        hf=True
    )
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-alpha")

    # Pipeline
    generator = pipeline(
        model=model, tokenizer=tokenizer,
        task='text-generation',
        max_new_tokens=50,
        repetition_penalty=1.2,
    )

    # Text generation with Zephyr
    zephyr = TextGeneration(generator, prompt=custom_prompts.REPRESENTATION_MODEL_PROMPT)
    representation_model = {"Zephyr": zephyr}
    return representation_model


if __name__ == "__main__":
    main()
    


