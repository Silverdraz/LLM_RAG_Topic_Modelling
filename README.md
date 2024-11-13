# Yang Sen Wei, Aaron - MAS Assignment Statement 2 (Minor) & Statement 3 (Major)
Statmenet 2 (Minor) was selected coupled with the mandatory Statement 3 question (Major)

## Setting Up
1. Create a conda environment with python environment 3.9 for both assignments
```
conda create -n env_name python=3.9 anaconda
```

2. Activate the conda environment and install pip in conda env
```
conda install pip
```

3. Install the libraries/frameworks using different requirements.txt for separate statements 2 and 3

4. For statement 3, ollama was used instead of HuggingFace as there is a bug in the required prompt formatting for different LLMs. ollama would have to be downloaded locally

5. For statement 3, download llama 3.2 locally by opening terminal and entering the command
```
ollama run llama3.2
```

## Answers to Statement 2
For statement 2, **to answer question 1**, figures stored under visualisations directory are used to understand the topics from the response text data. **To answer question 2**, department_topics.html figure was used to plot the topics concerning the different departments. **For the bonus question**, the representation model was used in an effort to understand the profile of individuals using the generated topics for some contextual information (keywords and representative documents) However, further efforts in prompt engineering may generate better results. These results are stored in the results folder

## Answers to Statement 3
For statement 3, **to answer question 1**, RAG was used as it is believed that financial documents are always updated and there will always be new documents. **To answer question 2**, appropriate retrieval methods (Multiquery retriever for better query analysis & Parent Document Retrieval for longer context using adjacent chunks) were utilised and were observed to retrieve appropriate context for the question. **For the bonus question**, the sources are appended to the text generation with doc and set of page numbers