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
import streamlit as st
import json
import logging



load_dotenv()
brwoserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")

logging.basicConfig(filename='my_log_file.txt',  # Name of the log file
                    level=logging.INFO,           # Set the logging level
                    format='%(asctime)s - %(levelname)s - %(message)s')

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
    logging.info('SEARCH: %s', response.text)
    return response.text
# search("Most popular prams for newborns")

# 2. Tool for scraping
def scrape_website(objective: str, url: str):
    # scrape website, and also will summarize the content based on objective if the content is too large
    # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    try:
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
                output = detailed_summary(objective, text)
                logging.info('SCRAPE WEBSITE SUMMARY: %s', output)
                return output
            else:
                logging.info('SCRAPE WEBSITE text: %s', text)
                return text
        else:
            print(f"HTTP request failed with status code {response.status_code}")
    except requests.exceptions.ConnectionError as e:
        print(f"Failed to establish a new connection: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary in JSON Format of the following text for {objective}:
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

def detailed_summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    
    # Define multiple prompts
    prompts = {
        "summary": "Write a comprehensive summary focusing on features and benefits for {objective}: {text}",
        "what_we_like": "Enumerate 1-3 points highlighting what stands out for {objective}: {text}",
        "best_for": "Specify the type of user or situation this product is ideally suited for {objective}: {text}",
        "price": "Extract and provide the specific price or price range of the product for {objective}, if available in the content: {text}",
        # "dimensions": "Provide any available dimensions like length, width, height for {objective}: {text}",
        # "weight": "Extract and mention the weight of the product if it is available for {objective}: {text}",
        "image": "If an image URL of the product is present, please extract and return it for {objective}: {text}",
        # "buy_link": "Extract and provide the direct link where this product can be purchased, if available in the content f{objective}: {text}"
}
    
    output = {}
    
    for info_type, prompt_template in prompts.items():
        map_prompt_template = PromptTemplate(
            template=prompt_template, input_variables=["text", "objective"])
        
        info_chain = load_summarize_chain(
            llm=llm,
            chain_type='map_reduce',
            map_prompt=map_prompt_template,
            combine_prompt=map_prompt_template,
            verbose=True
        )
        
        output[info_type] = info_chain.run(input_documents=docs, objective=objective)
        
    return json.dumps(output)  # Convert the output dictionary to JSON

# scrape_website("Chicco KeyFit 30","https://babygearessentials.com/chicco-keyfit-30/")

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
        logging.info('ScrapeWebsiteTool-BaseTool: %s', scrape_website(objective, url))
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
    content="""You are a world-class researcher, who can do detailed research on any topic and produce facts-based results; you do not make things up, you will try as hard as possible to gather facts & data to back up the research. You should research for top 10 best/most popular products base on the objective

            Please complete the following research objective while adhering to these rules:
            1/ You should do enough research to gather as much information as possible about the objective.
            2/ If there are URLs of relevant links & articles, please include them.
            3/ After gathering data, think, "Is there anything new I should search for based on the data I collected to increase research quality?" If the answer is yes, continue; But don't do this more than 3 iterations.
            4/ You should not make things up; you should only write facts & data that you have gathered.
            5/ In the final output, please return the research findings in a structured JSON format. The JSON should include a "research_summary", an array of "items" with details, and a "sources" array with reference links.
            6/ In the final output, please return the research findings in a structured JSON format. The JSON should include a "research_summary", an array of "items" with details, and a "sources" array with reference links.
            7/ In the final output, please return the research findings in a structured JSON format. The JSON should include a "research_summary", an array of "items" with details, and a "sources" array with reference links.
            8/ For example, if the research topic is 'Best Baby Car Seats for 2023', the JSON should look something like this:
            8/ {"research_summary":"Summary of the researçch...","items":[{"name":"Product Name","description":"Product Description","source":"URL Source","what_we_like":"List 1-3 points about what makes this product stand out","best_for":"Describe the type of user or situation this product is best suited for","price":"$0.00","dimension":"Provide dimensions such as length, width, height","weight":"weight","image":"image url","buy_link":"Extract and provide the direct link where this product can be purchased"}],"sources":[{"title":"Source Title","link":"Source Link"}]}
            10/ Please make sure the JSON is well-formatted and valid. Only returned one set of research_summary to avoid duplicate.
            11/ Please make sure the JSON is well-formatted and valid. Only returned one set of research_summary to avoid duplicate.
            12/ Please make sure the JSON is well-formatted and valid. Only returned one set of research_summary to avoid duplicate."""
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
def main():
    st.set_page_config(page_title="AI research agent", page_icon=":bird:")

    st.header("AI research agent :bird:")
    query = st.text_input("Research goal")

    if query:
        st.write("Doing research for ", query)

        result = agent({"input": query})

        st.info(result['output'])


if __name__ == '__main__':
    main()


# 5. Set this as an API endpoint via FastAPI
app = FastAPI()


class Query(BaseModel):
    query: str


@app.post("/")
def researchAgent(query: Query):
    query = query.query
    content = agent({"input": query})
    actual_content = content['output']

    # Step 2: Parse the string into a Python dictionary
    # parsed_dict = json.loads(actual_content.replace('\n', '').replace('\\', ''))

    return remove_duplicate_json(actual_content)



def remove_duplicate_json(json_str):

    json_str = json_str.replace('}}\n{{', '}\n{"')

    # Split the string by the delimiter '}{'
    json_list = json_str.split("}\n{")
    
    # Create an empty set to store unique JSON objects
    unique_json_set = set()
    
    # Iterate through the list and add each unique JSON object to the set
    for json_obj in json_list:
        # Add curly braces back to each JSON object
        if not json_obj.startswith("{"):
            json_obj = "{" + json_obj
        if not json_obj.endswith("}"):
            json_obj = json_obj + "}"
        
        unique_json_set.add(json_obj)
    
    # Join the unique JSON objects back into a single string
    unique_json_str = "}\n{".join(unique_json_set)
    
    print (unique_json_str)
    return unique_json_str

def remove_duplicate_research_summary(json_str_list):
    unique_research_dict = {}
    
    for json_str in json_str_list:
        parsed_json = json.loads(json_str)
        research_summary = parsed_json.get("research_summary")
        
        if research_summary not in unique_research_dict:
            unique_research_dict[research_summary] = parsed_json

    # Convert the unique dictionaries back to JSON strings
    unique_json_list = [json.dumps(item, indent=4) for item in unique_research_dict.values()]
    
    return unique_json_list

