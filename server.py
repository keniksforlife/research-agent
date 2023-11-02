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
import random
import time
import asyncio
from pyppeteer import connect, errors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from createJSON import transform_product_data
from io import StringIO
from typing import List, Dict, Any


load_dotenv()
brwoserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")
airtable_key = os.getenv("AIRTABLE_API_KEY")

# List of User Agents
user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 13_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15',
]

class ProductData(BaseModel):
    input_json: List[Dict[str, Any]] = Field(..., example=[{"Products": [], "Article Title": "The Best Prams of 2023", "Article ID": "recEDOOL9KxHsnHOl"}])


# HELPER FUNCTIONS
def clean_text(text):
    # Remove any kind of whitespace (including newlines and tabs)
    text = re.sub(r'\s+', ' ', text).strip()

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


# 1. Tool for search


def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity


def filter_similar_products(product_list, threshold=0.35):
    filtered_products = []
    for i, product1 in enumerate(product_list):
        is_similar = False
        for j, product2 in enumerate(filtered_products):
            similarity = calculate_similarity(
                product1['title'], product2['title'])
            # print(f"Comparing: {product1['title']} AND {product2['title']}. Similarity: {similarity}")
            if similarity > threshold:
                is_similar = True
                # print(f"Products are similar. Skipping: {product1['title']}")
                break
        if not is_similar:
            filtered_products.append(product1)
            # print(f"Adding product: {product1['title']}")
    return filtered_products


def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query + " site:amazon.com intext:/dp/",
        "num": 40
    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        # Raise an HTTPError if the HTTP request returned an unsuccessful status code
        response.raise_for_status()

        results_json = json.loads(response.text)
        # Retrieve the list of organic results
        organic_results = results_json.get('organic', [])

        amazon_results = [
            result for result in organic_results if 'amazon.com' in result.get('link', '')]

        print(
            f"Successfully fetched {len(amazon_results)} Amazon results.")

        # Filter out similar products
        unique_amazon_results = filter_similar_products(amazon_results)

        print(
            f"After filtering, {len(unique_amazon_results)} unique Amazon results remain.")

        return unique_amazon_results

    except requests.RequestException as e:
        logging.error(f"Failed to fetch search results: {e}")
        return None


# print(search("Best Pregnancy Pillows"))
# 2. Tool for scraping


async def scrape_website(objective: str, url: str):
    print("Start Scraping")
    # Connect to Browserless.io
    browser = await connect(browserWSEndpoint=f"wss://chrome.browserless.io?token={brwoserless_api_key}")

    # Create a new page
    page = await browser.newPage()

    # Randomly select a User-Agent
    selected_user_agent = random.choice(user_agents)

    # Set User-Agent
    await page.setUserAgent(selected_user_agent)

    # Navigate to the URL
    await page.goto(url)
    print("Visited: ", url)

    # Random sleep to mimic user reading page
    await asyncio.sleep(random.uniform(3, 7))

    try:
        await page.mouse.move(random.randint(0, 500), random.randint(0, 500))
    except pyppeteer.errors.NetworkError as e:
        print(f"An error occurred: {e}")

    # Random sleep before next action
    await asyncio.sleep(random.uniform(2, 5))

    # Scrape content and manipulate as needed
    content = await page.content()
    soup = BeautifulSoup(content, "html.parser")

    print(soup)
    try:
        name_elem = soup.select_one('span.product-title-word-break')
        price_elem = soup.select_one(
            '.reinventPricePriceToPayMargin.priceToPay')

        image_url_elem = soup.select_one('img[data-old-hires]')
        image_url_elem2 = soup.select_one(
            '[data-action="main-image-click"] img')
        details_elem = soup.select_one('div#detailBullets_feature_div')
        details_elem2 = soup.select_one('div#productDetails_feature_div')
        details_elem3 = soup.select_one('div#prodDetails')
        description_elem = soup.select_one('.a-spacing-small p span')
        about_elem = soup.select_one(
            'div.a-spacing-medium.a-spacing-top-small')

        # Check for None before accessing attributes
        name = str(name_elem.text.strip()) if name_elem else "N/A"

        details = clean_text(str(details_elem.text.strip())
                             ) if details_elem else "N/A"
        description = str(description_elem.text.strip()
                          ) if description_elem else "N/A"
        about = str(about_elem.text.strip()) if about_elem else "N/A"
        image_url = image_url_elem['data-old-hires'] if image_url_elem else "N/A"

        if (image_url == "N/A"):
            image_url = clean_text(
                str(image_url_elem2.text.strip())) if image_url_elem2 else "N/A"

        if (details == "N/A"):
            details = clean_text(
                str(details_elem2.text.strip())) if details_elem2 else "N/A"

        if (details == "N/A"):
            details = clean_text(
                str(details_elem3.text.strip())) if details_elem3 else "N/A"

        product_details = {
            "sp_name": name,
            "Images": [
                {
                    "url": is_valid_url(image_url)
                }
            ],
            "Image Link": is_valid_url(image_url),
            "sp_other_details": details,
            "sp_description": description,
            "sp_about": about,
            "Buy Link": url,
        }

        print(product_details)

        # Serialize the Python dictionary to a JSON-formatted string
        # product_details_json = json.dumps(product_details, ensure_ascii=False)

        if (name == "N/A"):
            return "Not a valid product content. Please find another product."
        else:
            return product_details

    except AttributeError as e:
        logging.error(f"Failed to scrape some attributes: {e}")

    await browser.close()

# asyncio.run(scrape_website("Best Moisturizers for dry skin","https://www.amazon.com/Cetaphil-DailyAdvance-Hydrating-Lotion-Sensitive/dp/B00EZWUHAM"))

class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(
        description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")


class ScrapeWebsiteTool(BaseTool):
    print("Start ScrapeWebsite Tool")
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        print("WEBSITE Tool: ", url)
        return asyncio.run(scrape_website(objective, url))

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
            2/ Return results that are directly related to the specific category or keyword. Do not include unrelated products.
            3/ Specifically, gather the direct Amazon URLs of these top 10 individual products. Only return URLs that include '/dp/' in the link, as these are direct product pages. Do not return URLs that lead to category or search result pages.
            4/ After each scraping iteration, evaluate if additional scraping rounds could improve the quality of your research. Limit this to no more than 3 iterations.
            5/ Only include facts & data gathered from Amazon's website. Do not make things up.
            6/ In the final output, include all reference data & direct product links to back up your research. Return the findings in a structured JSON format.
            7/ In the final output, include all reference data & direct product links to back up your research. Return the findings in a structured JSON format.
            8/ In the final output, include all reference data & direct product links to back up your research. Return the findings in a structured JSON format.
            9/ The JSON should look like this:
            {"research_summary": "Summary of the research...", "items": [{"name": "Product Name", "description": "product_description", "source": "Amazon URL", "what_we_like": "List 1-3 points about what makes this product stand out", "best_for": "Type of user or situation", "price": "Price of the product","in_the_box": "What's included in the box"}]}
            10/ The JSON should look like this:
            {"research_summary": "Summary of the research...", "items": [{"name": "Product Name", "description": "product_description", "source": "Amazon URL", "what_we_like": "List 1-3 points about what makes this product stand out", "best_for": "Type of user or situation", "price": "Price of the product", "in_the_box": "What's included in the box"}]}
            11/ The JSON should look like this:
            {"research_summary": "Summary of the research...", "items": [{"name": "Product Name", "description": "product_description", "source": "Amazon URL", "what_we_like": "List 1-3 points about what makes this product stand out", "best_for": "Type of user or situation", "price": "Price of the product", "in_the_box": "What's included in the box"}]}
            12/ If the output is not in JSON structured format, please update it and format in a structured JSON using the above template.
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


# 5. Set this as an API endpoint via FastAPI
app = FastAPI()


class Query(BaseModel):
    query: str


def long_running_task(query, unique_id, max_attempts=3):
    if max_attempts == 0:
        print("Maximum attempts reached. Could not get valid JSON.")
        return

    # content = agent({"input": "Top 10 " + query})
    search_results = search(query)

    if search_results is None:
        print("Search failed. Exiting.")
        return

    # Extract URLs from search results
    urls = [result['link'] for result in search_results]

    # Initialize an empty list to hold all the scraped data
    all_product_details = []

    # Counter for the number of products with image URLs
    product_count_with_images = 0

    # Create a dictionary for easy lookup of search result items by URL
    search_results_dict = {result['link']: result for result in search_results}

   # Step 2: Loop through each URL to scrape data.
    for url in urls:
        product_details = asyncio.run(
            scrape_website('Scrape product details', url))

        # Include search result data if available
        search_result_item = search_results_dict.get(url, {})
        price = search_result_item.get('price', 'N/A')
        snippet = search_result_item.get('snippet', 'N/A')

        if isinstance(product_details, dict):
            # Check if the product has an image URL
            if product_details.get('Images') and product_details['Images'][0].get('url').strip() not in ["", "N/A"]:
                product_details['Price'] = str(price)
                product_details['Description'] = snippet

                all_product_details.append(product_details)
                product_count_with_images += 1

                # Stop if we've gathered 10 products with image URLs
                if product_count_with_images >= 10:
                    break
        else:
            print("Warning: product_details is not a dictionary",
                  type(product_details))
            print("product details: ", product_details)

    actual_content = all_product_details

    try:
        if (is_valid_json):
            save_to_airtable(remove_duplicate_json(
                actual_content), query, unique_id)
        else:
            print(f"Invalid JSON received. Attempts left: {max_attempts - 1}")
            long_running_task(query, unique_id, max_attempts=max_attempts - 1)
    except Exception as e:
        print(f"An error occurred: {e}")


@app.post("/")
def researchAgentV2(query: Query):
    query = query.query
    unique_id = str(uuid.uuid4())
    # Start a new thread for the long-running task
    thread = Thread(target=long_running_task, args=(query, unique_id))
    thread.start()

    return {"message": "Request is being processed", "id": unique_id}

@app.post("/createJSON")
async def create_json(data: ProductData):
    # Convert input JSON data to string
    input_json_str = json.dumps(data.input_json)
    
    # Use StringIO to simulate a file-like object for input JSON data
    input_json_file = StringIO(input_json_str)
    
    # Use StringIO to create a file-like object for output JSON data
    output_json_file = StringIO()
    
    # Call the transform_product_data function
    transform_product_data(input_json_file, output_json_file)
    
    # Get the transformed JSON data from the output file-like object
    output_json_file.seek(0)
    output_json_data = json.load(output_json_file)
    
    return output_json_data


def save_to_airtable(all_product_details, category, unique_id):
    API_URL = "https://api.airtable.com/v0/appMIkd5mMSKDXzkr/Products"

    print("SAVING RECORDS TO AIRTABLE . . .")
    headers = {
        "Authorization": f"Bearer {airtable_key}",
        "Content-Type": "application/json"
    }

    # Initialize an empty list to hold the data dictionaries
    data_list = []

    # Loop through all items in the parsed_json
    for item in all_product_details:
        data_dict = {
            "fields": {
                "Product Name": item.get('sp_name'),
                "Source": item['Buy Link'],
                "Category": category,
                "batch_id": unique_id
            }
        }
        # Append the data_dict to the data_list
        data_list.append(data_dict)
        data_dict['fields'].update(item)

    response = requests.post(API_URL, headers=headers, json={
        "records": data_list})
    if response.status_code != 200:
        print(f"Failed to add record: {response.content}")

    if response.status_code == 200:

        # Create nre record with batch_id in Generate Content Table
        API_URL = "https://api.airtable.com/v0/appMIkd5mMSKDXzkr/Generated%20Articles"

        data = {
            "fields": {
                "batch_id": unique_id
            }}

        requests.post(API_URL, headers=headers, json=data)

        # Execute Generation of Artile Scenario in Make
        API_URL = "https://hook.eu1.make.com/5uyqhpqm1beskwadyysebuvq23na7734"

        data = {
            "batch_id": unique_id
        }

        requests.get(API_URL, json=data)

        print("Record successfully added.")
    else:
        print(f"Failed to add record: {response.content}")
