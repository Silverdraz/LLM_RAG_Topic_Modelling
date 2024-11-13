"""
Preprocesses the dataframes, such as removing not applicable comments or NA comments (placeholder comments) which would definitely not
 be considered a topic and can instead be noise, affecting the predictive performance of the BERTopic Model.
"""

def remove_na_comments(data):
    """Remove redundant rows that are simply not applicable or -- placeholder comments

    Args:
        data: pandas dataframe of the employee engagement survey

    Returns:
        data: preprocessed pandas dataframe removing rows consisting of not applicable or placeholder comments
    """    
    #Remove all possible placeholder comments that are "noise" and do not contribute to predictive performance
    data = data.drop(data[(data['employee_feedback'].str.lower() == "na") | 
                          (data['employee_feedback'].str.lower().str.strip() == "n.a.") |
                          (data['employee_feedback'].str.lower() == "none") |
                          (data['employee_feedback'].str.lower().str.strip() == "nil") |
                          (data['employee_feedback'].str.lower() == "no") |
                          (data['employee_feedback'].str.lower() == "nil.") |
                          (data['employee_feedback'].str.lower() == "na.") |
                          (data['employee_feedback'].str.lower() == "n.a.") |
                          (data['employee_feedback'].str.lower() == "md") |
                          (data['employee_feedback'].str.lower() == "-") |
                          (data['employee_feedback'].str.lower() == "--")].index).reset_index()
    return data