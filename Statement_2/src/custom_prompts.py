"""
Custom prompts for the representation model prompts. Consolidated here in this script for convenience and consistency in case there are multiple
representation or text generation prompts
"""


#System prompt for use in the LLM Chat in genaipipe.py
REPRESENTATION_MODEL_PROMPT = """<|system|>You are a helpful, respectful and honest assistant for labeling topics..</s>
<|user|>
I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Based on the information about the topic above, please describe the profile of the employees. Make sure you describe the
profile of the employees only using the documents and keywords above.</s>
<|assistant|>"""

