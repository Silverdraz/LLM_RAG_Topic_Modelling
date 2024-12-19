## Setting Up
1. Create a conda environment with python environment 3.9 
```
conda create -n env_name python=3.9 anaconda
```

2. Activate the conda environment and install pip in conda env
```
conda install pip
```

3. Install the libraries/frameworks using different requirements.txt 

4. For statement 3, ollama was used instead of HuggingFace as there is a bug in the required prompt formatting for different LLMs. ollama would have to be downloaded locally

5. For statement 3, download llama 3.2 locally by opening terminal and entering the command
```
ollama run llama3.2
```
