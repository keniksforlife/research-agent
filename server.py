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
import logging
import re
from langchain.schema import SystemMessage
from fastapi import FastAPI
from threading import Thread
import streamlit as st
from openai.error import InvalidRequestError
import uuid
from urllib.parse import urlparse
from json import JSONDecodeError

load_dotenv()
brwoserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")

all_product_details = []

# 1. Tool for search


def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query + " site:amazon.com intext:/dp/"
    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        response.raise_for_status()  # Raise an HTTPError if the HTTP request returned an unsuccessful status code

        results_json = json.loads(response.text)
        organic_results = results_json.get('organic', [])  # Retrieve the list of organic results

        amazon_results = [result for result in organic_results if 'amazon.com' in result.get('link', '')]

        logging.info(f"Successfully fetched {len(amazon_results)} Amazon results.")
        return amazon_results

    except requests.RequestException as e:
        logging.error(f"Failed to fetch search results: {e}")
        return None


# 2. Tool for scraping

def clean_text(text):
    # Remove any kind of whitespace (including newlines and tabs)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Replace newline characters with commas
    # text = text.replace("\n", "\n")
    
    return text

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
# def scrape_website(objective: str, url: str):
#     # scrape website, and also will summarize the content based on objective if the content is too large
#     # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

#     print("Scraping website...")
#     # Define the headers for the request
#     headers = {
#         'Cache-Control': 'no-cache',
#         'Content-Type': 'application/json',
#     }

#     # Define the data to be sent in the request
#     data = {
#         "url": url
#     }

#     # Convert Python object to JSON string
#     data_json = json.dumps(data)

#     # Send the POST request
#     post_url = f"https://chrome.browserless.io/content?token={brwoserless_api_key}"
#     response = requests.post(post_url, headers=headers, data=data_json)

#     # Check the response status code
#     if response.status_code == 200:
#         soup = BeautifulSoup(response.content, "html.parser")
#         text = soup.get_text()
#         print("CONTENTTTTTT:", text)

#         if len(text) > 10000:
#             output = summary(objective, text)
#             return output
#         else:
#             return text
#     else:
#         print(f"HTTP request failed with status code {response.status_code}")


def scrape_website(objective: str, url: str):
    # scrape website, and also will summarize the content based on objective if the content is too large
    # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    print("Scraping Amazon website...")
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
        # text = soup.get_text()
        # print("CONTENTTTTTT:", text)

        try:
            name_elem = soup.select_one('span.product-title-word-break')
            # price_elem = soup.select_one('.reinventPricePriceToPayMargin.priceToPay')
            image_url_elem = soup.select_one('[data-action="main-image-click"] img')
            image_url_elem1 = soup.select_one('.itemNo1 [data-action="main-image-click"] img')
            image_url_elem2 = soup.select_one('.itemNo2 [data-action="main-image-click"] img')
            image_url_elem3 = soup.select_one('.itemNo3 [data-action="main-image-click"] img')
            image_url_elem4 = soup.select_one('.itemNo4 [data-action="main-image-click"] img')
            image_url_elem5 = soup.select_one('.itemNo5 [data-action="main-image-click"] img')
            details_elem = soup.select_one('div#detailBullets_feature_div')
            details_elem2 = soup.select_one('div#productDetails_feature_div')
            details_elem3 = soup.select_one('div#prodDetails')
            description_elem = soup.select_one('.a-spacing-small p span')
            about_elem = soup.select_one('div.a-spacing-medium.a-spacing-top-small')

            # Check for None before accessing attributes
            name = str(name_elem.text.strip()) if name_elem else "N/A"
    
            image_url = str(image_url_elem['src'].strip()) if image_url_elem else "N/A"
            image_url1 = str(image_url_elem1['src'].strip()) if image_url_elem1 else "N/A"
            image_url2 = str(image_url_elem2['src'].strip()) if image_url_elem2 else "N/A"
            image_url3 = str(image_url_elem3['src'].strip()) if image_url_elem3 else "N/A"
            image_url4 = str(image_url_elem4['src'].strip()) if image_url_elem4 else "N/A"
            image_url5 = str(image_url_elem5['src'].strip()) if image_url_elem5 else "N/A"
            details = clean_text(str(details_elem.text.strip())) if details_elem else "N/A"
            description = str(description_elem.text.strip()) if description_elem else "N/A"
            about = str(about_elem.text.strip()) if about_elem else "N/A"

            if(details == "N/A"):
                details = clean_text(str(details_elem2.text.strip())) if details_elem2 else "N/A"

            if(details == "N/A"):
                details = clean_text(str(details_elem3.text.strip())) if details_elem3 else "N/A"

            product_details = {
                "sp_name": name,
                "sp_images": [
                    {
                        "url": is_valid_url(image_url)
                    }
                ],
                "sp_other_details": details,
                "sp_description": description,
                "sp_about": about,
                "Buy Link": url,
            }

            # Serialize the Python dictionary to a JSON-formatted string
            # product_details_json = json.dumps(product_details, ensure_ascii=False)

            # all_product_details.append(product_details)

            if (name == "N/A"):
                return "Not a valid product content. Please find another product."
            else:
                return product_details
        

        except AttributeError as e:
            logging.error(f"Failed to scrape some attributes: {e}")

    else:
        print(f"HTTP request failed with status code {response.status_code}")

# ast = scrape_website("Scrape product details","https://www.amazon.com/Mommys-Bliss-Organic-Soothing-Massage/dp/B08643YT9K")
# print(ast)

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
    content="""You are a world-class researcher, focused on gathering detailed and factual information solely from Amazon's website; 
            you do not make things up, you will strive to gather facts & data to back up the research;
            
            Please complete the objective above with the following rules:
            1/ You should scrape Amazon's website to gather as much information as possible about the top instock 10 best/most popular products in a given category. Please ignore out of stock products
            2/ Specifically, gather the direct Amazon URLs of these top 10 individual products. Only return URLs that include '/dp/' in the link, as these are direct product pages. Do not return URLs that lead to category or search result pages.
            3/ After each scraping iteration, evaluate if additional scraping rounds could improve the quality of your research. Limit this to no more than 3 iterations.
            4/ Only include facts & data gathered from Amazon's website. Do not make things up.
            5/ In the final output, include all reference data & direct product links to back up your research. Return the findings in a structured JSON format.
            6/ In the final output, include all reference data & direct product links to back up your research. Return the findings in a structured JSON format.
            7/ In the final output, include all reference data & direct product links to back up your research. Return the findings in a structured JSON format.
            8/ The JSON should look like this:
            {"research_summary": "Summary of the research...", "items": [{"name": "Product Name", "description": "product_description", "source": "Amazon URL", "what_we_like": "List 1-3 points about what makes this product stand out", "best_for": "Type of user or situation", "price": "Price of the product", "image": "product_image_url", "dimensions": "product_other_details", "weight": "Weight data", "in_the_box": "What's included in the box"}]}
            9/ The JSON should look like this:
            {"research_summary": "Summary of the research...", "items": [{"name": "Product Name", "description": "product_description", "source": "Amazon URL", "what_we_like": "List 1-3 points about what makes this product stand out", "best_for": "Type of user or situation", "price": "Price of the product", "image": "product_image_url", "dimensions": "product_other_details", "weight": "Weight data", "in_the_box": "What's included in the box"}]}
            10/ The JSON should look like this:
            {"research_summary": "Summary of the research...", "items": [{"name": "Product Name", "description": "product_description", "source": "Amazon URL", "what_we_like": "List 1-3 points about what makes this product stand out", "best_for": "Type of user or situation", "price": "Price of the product", "image": "product_image_url", "dimensions": "product_other_details", "weight": "Weight data", "in_the_box": "What's included in the box"}]}
            11/ If the output is not in JSON structured format, please update it and format in a structured JSON using the above template.
            """
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

def long_running_task(query,unique_id, max_attempts=3):
    if max_attempts == 0:
        print("Maximum attempts reached. Could not get valid JSON.")
        return
    
    content = agent({"input": "Top 10 " + query})
    actual_content = content['output']
    print("actual %s",actual_content)

    if(is_valid_json):
        save_to_airtable(remove_duplicate_json(actual_content), query, unique_id)
    else:
        print(f"Invalid JSON received. Attempts left: {max_attempts - 1}")
        long_running_task(query, unique_id, max_attempts=max_attempts - 1)


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

def is_valid_json(json_str):
    try:
        json.loads(json_str)
        return True
    except JSONDecodeError:
        return False


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

        product_details = scrape_website("Scrape product details",item['source'])
        print("product_details", product_details)

        if isinstance(product_details, dict):
            if (product_details.get('sp_name') != "N/A"):
                data_dict['fields'].update(product_details)
        else:
            print("Warning: product_details is not a dictionary. Skipping update.")

        if len(data_list) >= 10:
            response = requests.post(API_URL, headers=headers, json={"records": data_list})
            if response.status_code != 200:
                print(f"Failed to add record: {response.content}")
            data_list.clear()  # Clear the list for the next batch

    print("new json %s",json.dumps(data_list))
 
    # Send any remaining records that are less than 10
    if len(data_list) > 0:
        response = requests.post(API_URL, headers=headers, json={"records": data_list})
        if response.status_code != 200:
            print(f"Failed to add record: {response.content}")


    if response.status_code == 200:

        #Create record in Generate Content Table
        API_URL = "https://api.airtable.com/v0/appMIkd5mMSKDXzkr/Generated%20Articles"

        data = {
            "fields": {
                "batch_id": unique_id
            }}

        requests.post(API_URL, headers=headers, json=data)

         #Create record in Generate Content Table
        API_URL = "https://hook.eu1.make.com/927ubkqsww2puh5uwhg6ong2fppw0t59"

        data = {
                "batch_id": unique_id
            }

        requests.get(API_URL, json=data)

        print("Record successfully added.")
    else:
        print(f"Failed to add record: {response.content}")
