#use dotenv to get the api keys from .env file
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())




#Import prompt and define PromptTemplate
from langchain import PromptTemplate
template = """
You are an expert data scientist with an expertise in building deep learning models. 
Explain the concept of {concept} in a couple of lines
"""
prompt = PromptTemplate(
    input_variables=["concept"],
    template=template,
)




#Run LLM with PromptTemplate
from langchain.llms import OpenAI
llm = OpenAI(model_name="text-davinci-003")
# print(llm(prompt.format(concept="autoencoder")))