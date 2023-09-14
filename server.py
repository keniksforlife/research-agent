import os
from dotenv import load_dotenv

from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
from langchain.schema import SystemMessage
from fastapi import FastAPI
from threading import Thread
import streamlit as st
from openai.error import InvalidRequestError
import json
import uuid
from urllib.parse import urlparse

load_dotenv()
brwoserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")

# 1. Tool for search


def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query
    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)

    return response.text


# 2. Tool for scraping
def scrape_website(objective: str, url: str):
    # scrape website, and also will summarize the content based on objective if the content is too large
    # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    print("Scraping website...")
    # Define the headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Define the data to be sent in the request
    data = {
        "url": url
    }

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request
    post_url = f"https://chrome.browserless.io/content?token={brwoserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)

    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENTTTTTT:", text)

        if len(text) > 10000:
            output = summary(objective, text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")


def summary(objective, content):
    try:
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
        docs = text_splitter.create_documents([content])
        map_prompt = """
        Write a summary of the following text for {objective}:
        "{text}"
        SUMMARY:
        """
        map_prompt_template = PromptTemplate(
            template=map_prompt, input_variables=["text", "objective"])

        summary_chain = load_summarize_chain(
            llm=llm,
            chain_type='map_reduce',
            map_prompt=map_prompt_template,
            combine_prompt=map_prompt_template,
            verbose=True
        )

        output = summary_chain.run(input_documents=docs, objective=objective)

        return output
    except InvalidRequestError as e:
        # Handle the error gracefully
        print(f"Error: {e}")
        return None

class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(
        description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")


class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")


# 3. Create langchain agent with the tools above
tools = [
    Tool(
        name="Search",
        func=search,
        description="useful for when you need to answer questions about current events, data. You should ask targeted questions"
    ),
    ScrapeWebsiteTool(),
]

system_message = SystemMessage(
    content="""You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
            you do not make things up, you will try as hard as possible to gather facts & data to back up the research;
            You should research for the top 10 best/most popular products base on the objective
            
            Please make sure you complete the objective above with the following rules:
            1/ You should do enough research to gather as much information as possible about the objective
            2/ If there are url of relevant links & articles, you will scrape it to gather more information
            3/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iteratins
            4/ You should not make things up, you should only write facts & data that you have gathered
            5/ In the final output, You should include all reference data & links to back up your research and return the research findings in a structured JSON format; You should include all reference data & links to back up your research and return the research findings in a structured JSON format
            6/ In the final output, You should include all reference data & links to back up your research and return the research findings in a structured JSON format; You should include all reference data & links to back up your research and return the research findings in a structured JSON format
            7/ The JSON should look something like this:
            {"research_summary": "Summary of the research...", "items": [{"name": "Product Name", "description": "Product Description", "source": "URL Source / reference link", "what_we_like": "List 1-3 points about what makes this product stand out", "best_for": "Type of user or situation", "price": "Price of the product", "image": "Image URL in HD", "dimensions": "Dimensions data", "weight": "Weight data", "in_the_box": "What's included in the box"}]}
            8/ The JSON should look something like this:
            {"research_summary": "Summary of the research...", "items": [{"name": "Product Name", "description": "Product Description", "source": "URL Source / reference link", "what_we_like": "List 1-3 points about what makes this product stand out", "best_for": "Type of user or situation", "price": "Price of the product", "image": "Image URL in HD", "dimensions": "Dimensions data", "weight": "Weight data", "in_the_box": "What's included in the box"}]}
            9/ The JSON should look something like this:
            {"research_summary": "Summary of the research...", "items": [{"name": "Product Name", "description": "Product Description", "source": "URL Source / reference link", "what_we_like": "List 1-3 points about what makes this product stand out", "best_for": "Type of user or situation", "price": "Price of the product", "image": "Image URL in HD", "dimensions": "Dimensions data", "weight": "Weight data", "in_the_box": "What's included in the box"}]}"""
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)


# 4. Use streamlit to create a web app
# def main():
#     st.set_page_config(page_title="AI research agent", page_icon=":bird:")

#     st.header("AI research agent :bird:")
#     query = st.text_input("Research goal")

#     if query:
#         st.write("Doing research for ", query)

#         result = agent({"input": query})

#         st.info(result['output'])


# if __name__ == '__main__':
#     main()


# 5. Set this as an API endpoint via FastAPI
app = FastAPI()


class Query(BaseModel):
    query: str

def long_running_task(query,unique_id):
    content = agent({"input": query})
    actual_content = content['output']
    save_to_airtable(remove_duplicate_json(actual_content), query, unique_id)


@app.post("/v2")
def researchAgent(query: Query):
    query = query.query
    content = agent({"input": query})
    actual_content = content['output']
    return remove_duplicate_json(actual_content)

@app.post("/")
def researchAgentV2(query: Query):
    query = query.query
    unique_id = str(uuid.uuid4())
     # Start a new thread for the long-running task
    thread = Thread(target=long_running_task, args=(query,unique_id))
    thread.start()
    
    return {"message": "Request is being processed","id" : unique_id}

def remove_duplicate_json(json_str):

    # Split the string by the delimiter '}{'
    if "}\n{" in json_str:
        json_list = json_str.split("}\n{")
        return json_list[0] + "}"
    
    return json_str

def is_valid_url(url):

    try:
        # Parse the URL
        parsed_url = urlparse(url)
        
        # Check the components
        if all([parsed_url.scheme, parsed_url.netloc]):
            return url
        else:
            return ""
    except Exception as e:
        # Log the exception (in a real-world application)
        print(f"An error occurred: {e}")
        return ""

def save_to_airtable(json_str,category, unique_id):
    API_URL = "https://api.airtable.com/v0/appMIkd5mMSKDXzkr/Products"
    API_KEY = "patCKWLwcI38V3ls7.80c84f95c7b36e4cb14bc0453b22445f043bfbfef5f4a2c88c7d113a4921b56f"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # Parse the JSON string to a Python object
    parsed_json = json.loads(json_str)

    # Initialize an empty list to hold the data dictionaries
    data_list = []
    # unique_id = str(uuid.uuid4())

    # Loop through all items in the parsed_json
    for item in parsed_json['items']:
        data_dict = {
            "fields": {
                "Product Name": item['name'],
                "Source": item['source'],
                "Category": category,
                "Price": item['price'],  
                "What We Like": item['what_we_like'],
                "Description": item['description'],
                "Images": [
                    {
                        "url": is_valid_url(item['image'])
                    }
                ],
                "Best For": item['best_for'], 
                "In the box": item['in_the_box'], 
                "batch_id": unique_id
            }
        }
        # Append the data_dict to the data_list
        data_list.append(data_dict)
    print("new json %s",json.dumps(data_list))
 
    response = requests.post(API_URL, headers=headers, json={ "records":data_list })


    if response.status_code == 200:

        #Create record in Generate Content Table
        API_URL = "https://api.airtable.com/v0/appMIkd5mMSKDXzkr/Generated%20Articles"

        data = {
            "fields": {
                "batch_id": unique_id
            }}

        requests.post(API_URL, headers=headers, json=data)

        print("Record successfully added.")
    else:
        print(f"Failed to add record: {response.content}")
