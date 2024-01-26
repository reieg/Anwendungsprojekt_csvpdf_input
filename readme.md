This Repo is based on another public repository from alejandro-ao: https://github.com/alejandro-ao/langchain-ask-pdf

app.py extents the functionality of alejandro's programm in order to be able iterate through multiple files. In addition, pdf files and csv files can be uploaded and are eventually given as an input to the llm.


## Installation

To install the repository, please clone this repository and install the requirements:

```
pip install -r requirements.txt
```

You will also need to add your OpenAI API key to the `.env` file.

## usage
'''
streamlit run app.py
'''