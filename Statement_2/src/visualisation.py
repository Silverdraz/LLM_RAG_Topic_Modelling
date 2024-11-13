"""
Create plots for inspection to have a better understanding of the topics distribution across the data. Having common functions establish convenience 
since the naming of the files is the only different parameter
"""

import os #os file paths

#Global Constants
SAVE_VISUALISATION_PATH = r"..\visualisations" #Path to visualisation folder

def plot_heatmap(topic_model, file_name):
    """plot the heatmap to compare similarity across the topics

    Args:
        topic_model: trained topic model with learnt topics
        file_name: custom file name for this specific image
    """    
    heatmap_fig = topic_model.visualize_heatmap()
    heatmap_fig.write_html(os.path.join(SAVE_VISUALISATION_PATH,file_name))


def plot_document(docs,topic_model, file_name):
    """plot the documents with topics as legends or hue

    Args:
        docs: docs consisting of the text for the topic modelling
        topic_model: trained topic model with learnt topics
        file_name: custom file name for this specific image
    """    
    document_fig = topic_model.visualize_documents(docs)
    document_fig.write_html(os.path.join(SAVE_VISUALISATION_PATH,file_name))

def plot_data_document(docs,topic_model, file_name):
    """plot the documents with topics 

    Args:
        docs: docs consisting of the text for the topic modelling
        topic_model: trained topic model with learnt topics
        file_name: custom file name for this specific image
    """    
    data_document_fig = topic_model.visualize_document_datamap(docs)
    data_document_fig.savefig(os.path.join(SAVE_VISUALISATION_PATH,file_name))

def plot_topics_dept(topics_per_class, topic_model, file_name):
    """Plot the topics for each department subcategory

    Args:
        topics_per_class: Topics for each department
        topic_model: trained topic model with learnt topics
        file_name: custom file name for this specific image
    """    
    sub_category_fig = topic_model.visualize_topics_per_class(topics_per_class, top_n_topics=10)
    sub_category_fig.write_html(os.path.join(SAVE_VISUALISATION_PATH,file_name))