# Jumpstart your Machine Learning experiments

🧙&nbsp; A web app to dynamically generate template code for machine learning.


Generate custom template code for PyTorch & sklearn, using a simple web UI built with [streamlit](https://www.streamlit.io/). ML-Jumpstart offers multiple options for preprocessing, model setup, training, and visualization (using MLFlow). It exports to .py, Jupyter Notebook, or  [Google Colab](https://colab.research.google.com/). The perfect tool to jumpstart your next machine learning project! ✨ 

**Note: The steps below are only required for developers who want to run/deploy traingenerator locally.**

## Requirements

Python 3.7+

## Installation

Install the required packages in your local environment (ideally virtualenv, conda, etc.).

```bash
git clone https://github.com/shipt/ml-jumpstart.git
cd ml-jumpstart
pip install -r requirements.txt
``` 

## Run It

1. Start your  app with: 
```bash
streamlit run app/main.py
```

## Testing

```bash
pytest ./tests
```

This generates Python codes with different configurations (just like the app would do) 
and checks that they run. The streamlit app itself is not tested at the moment.
