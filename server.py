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
# from openai.error import InvalidRequestError
import uuid
from urllib.parse import urlparse
from json import JSONDecodeError
import random
import time
import asyncio
import pyppeteer
# from pyppeteer import connect, errors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from createJSON import transform_product_data
from io import StringIO
from typing import List, Dict, Any
import base64
import time


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
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:104.0) Gecko/20100101 Firefox/104.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:103.0) Gecko/20100101 Firefox/103.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:104.0) Gecko/20100101 Firefox/104.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:103.0) Gecko/20100101 Firefox/103.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Edge/109.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Edge/108.0.0.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/604.1.38 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 12; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 11; Pixel 4 XL) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Trident/7.0; rv:11.0) like Gecko",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36 OPR/85.0.4341.75",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15 OPR/85.0.4341.75"
]


class ProductData(BaseModel):
    input_json: List[Dict[str, Any]] = Field(..., example=[{"Products": [
    ], "Article Title": "The Best Prams of 2023", "Article ID": "recEDOOL9KxHsnHOl"}])


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

# Function to save cookies to a file
async def save_cookies(cookies, path="cookies.json"):
    with open(path, "w") as file:
        json.dump(cookies, file)

# Function to load cookies from a file


async def load_cookies(path="cookies.json"):
    try:
        with open(path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return None  # No cookies file found

api_key = '0fc58242b7ce53ffe32fb48490d580af'


def solve_captcha(captcha_image_url):
    # Download the image from the URL
    response = requests.get(captcha_image_url)
    print(captcha_image_url)
    if response.status_code == 200:
        # Convert the image to a base64-encoded string
        print("convertng image url to base64")
        encoded_image = base64.b64encode(response.content).decode('utf-8')
        # Send CAPTCHA for solving
        data = {
            'key': api_key,
            'method': 'base64',
            'body': encoded_image,  # Base64 encoded image
            'json': 1
        }
        response = requests.post('http://2captcha.com/in.php', data=data)
        request_id = response.json()['request']

        # Poll for the solved CAPTCHA
        for i in range(10):
            time.sleep(5)  # Wait for 5 seconds before each check
            result = requests.get(
                f'http://2captcha.com/res.php?key={api_key}&action=get&id={request_id}&json=1')
            print('solving ...')
            if result.json()['status'] == 1:
                # CAPTCHA Solved
                print('captcha solved')
                return result.json()['request']
    else:
        print(f"Failed to download CAPTCHA image: HTTP {response.status_code}")
    return None

# Example usage
# captcha_solution = solve_captcha("https://images-na.ssl-images-amazon.com/captcha/twhhswbk/Captcha_ysxgzjhfwo.jpg", api_key)
# if captcha_solution:
#     print("CAPTCHA Solved:", captcha_solution)
# else:
#     print("Failed to solve CAPTCHA")



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

async def search_amazon(query):
    amazon_url = "https://www.amazon.com/s?k=" + query

    # Connect to Browserless.io
    browser = await pyppeteer.connect(browserWSEndpoint=f"wss://chrome.browserless.io?token={brwoserless_api_key}")
    # Create a new page
    page = await browser.newPage()

    # Randomly select a User-Agent
    selected_user_agent = random.choice(user_agents)

    # Set User-Agent
    print('set useragent')
    await page.setUserAgent(selected_user_agent)

    print('load coockies')
    # Load cookies from file and set them if they exist
    cookies = await load_cookies()
    if cookies:
        await page.setCookie(*cookies)

    print('navigating', amazon_url)
    # Navigate to the URL
    await page.goto(amazon_url)
    print("Visited: ", amazon_url)

    # Retrieve and save cookies
    cookies = await page.cookies()
    await save_cookies(cookies)

    # Check if CAPTCHA is present
    print("Checking Captcha")

    try:
          # Scrape content and manipulate as needed
        content = await page.content()
        soup = BeautifulSoup(content, "html.parser")

        try:
            await asyncio.sleep(random.uniform(3, 15))

            captcha_image = await page.querySelector('img[src*="captcha"]')
            print(captcha_image)
            if captcha_image:
                print("Solving the captcha ...")
                captcha_image_url = await page.evaluate('(captcha_image) => captcha_image.src', captcha_image)

                # Solve CAPTCHA
                captcha_solution = solve_captcha(captcha_image_url)
                print("captcha status", captcha_solution)

                # Input the solution and submit the form
                if captcha_solution is not None and captcha_solution != "":
                    await page.type('#captchacharacters', captcha_solution)
                    await page.click('button[type="submit"]')
                    print('captcha submitted')
                    try:
                        # Wait for navigation to complete
                        await page.waitForNavigation()  # Timeout in milliseconds

                    except pyppeteer.errors.TimeoutError:
                        logging.error(
                            "Navigation timeout after CAPTCHA submission.")
                        # Handle timeout
                        # return "Navigation timeout occurred."
                else:
                    print("Captcha solution is not available or is invalid.")
        except pyppeteer.errors.NetworkError as e:
            print(f"No Captcha: {e}")


        search_results = soup.find_all('div', {'data-component-type': 's-search-result'})

        product_details = []
        for item in search_results:
            # Extract only the URL of the product detail page
            link_element = item.find('a', {'class': 'a-link-normal s-no-outline'}, href=True)
            price_element = item.find('span', {'class': 'a-price'})
            if link_element and 'dp/' in link_element['href']:
                product_url = 'https://www.amazon.com' + link_element['href']
                price = price_element.find('span', {'class': 'a-offscreen'}).text if price_element else "N/A"

                product_details.append({'url': product_url, 'price': price})


        print(f"Found {len(product_details)} product URLs")
        print(product_details)
        return product_details

    except pyppeteer.errors.NetworkError as e:
        print(f"Error during requests to {amazon_url} : {e}")
        return None
    finally:
        try:
            await browser.close()  # Ensure this is awaited
        except Exception as e:
            logging.error(f"Error closing browser: {e}")


# asyncio.run(search_amazon("high chair"))
# 2. Tool for scraping

async def scrape_website(objective: str, url: str):

    try:
        print("Start Scraping")
        # Connect to Browserless.io
        browser = await pyppeteer.connect(browserWSEndpoint=f"wss://chrome.browserless.io?token={brwoserless_api_key}")
        print('new page')
        # Create a new page
        page = await browser.newPage()

        # Randomly select a User-Agent
        selected_user_agent = random.choice(user_agents)

        # Set User-Agent
        print('set useragent')
        await page.setUserAgent(selected_user_agent)

        print('load coockies')
        # Load cookies from file and set them if they exist
        cookies = await load_cookies()
        if cookies:
            await page.setCookie(*cookies)

        print('navigating', url)
        # Navigate to the URL
        await page.goto(url)
        print("Visited: ", url)

        # Retrieve and save cookies
        cookies = await page.cookies()
        await save_cookies(cookies)

        # Check if CAPTCHA is present
        print("Checking Captcha")

        try:
            await asyncio.sleep(random.uniform(3, 15))

            captcha_image = await page.querySelector('img[src*="captcha"]')
            print(captcha_image)
            if captcha_image:
                print("Solving the captcha ...")
                captcha_image_url = await page.evaluate('(captcha_image) => captcha_image.src', captcha_image)

                # Solve CAPTCHA
                captcha_solution = solve_captcha(captcha_image_url)
                print("captcha status", captcha_solution)

                # Input the solution and submit the form
                if captcha_solution is not None and captcha_solution != "":
                    await page.type('#captchacharacters', captcha_solution)
                    await page.click('button[type="submit"]')
                    print('captcha submitted')
                    try:
                        # Wait for navigation to complete
                        await page.waitForNavigation()  # Timeout in milliseconds

                    except pyppeteer.errors.TimeoutError:
                        logging.error(
                            "Navigation timeout after CAPTCHA submission.")
                        # Handle timeout
                        # return "Navigation timeout occurred."
                else:
                    print("Captcha solution is not available or is invalid.")
        except pyppeteer.errors.NetworkError as e:
            print(f"No Captcha: {e}")

        # Random sleep to mimic user reading page
        await asyncio.sleep(random.uniform(3, 15))

        try:
            await page.mouse.move(random.randint(0, 500), random.randint(0, 500))
        except pyppeteer.errors.NetworkError as e:
            print(f"An error occurred: {e}")

        # Random sleep before next action
        await asyncio.sleep(random.uniform(2, 5))

        # Scrape content and manipulate as needed
        content = await page.content()
        soup = BeautifulSoup(content, "html.parser")

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

    except pyppeteer.errors.NetworkError as e:
        logging.error(f"NetworkError occurred: {e}")
        # Handle the network error (e.g., retry, log, exit, etc.)
        return "Network error occurred."

    except pyppeteer.errors.PageError as e:
        logging.error(f"PageError occurred: {e}")
        # Handle page errors (e.g., retry, log, exit, etc.)
        return "Page error occurred."

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        # Handle other unexpected errors
        return "An unexpected error occurred."

    finally:
        try:
            await browser.close()  # Ensure this is awaited
        except Exception as e:
            logging.error(f"Error closing browser: {e}")

# asyncio.run(scrape_website("","https://www.amazon.com/Nuby-Natural-Soothing-Benzocaine-Belladonna/dp/B079QLR1YX"))


# 5. Set this as an API endpoint via FastAPI
app = FastAPI()


class Query(BaseModel):
    query: str
    type: str


def long_running_task(query, unique_id, type, max_attempts=3):
    if max_attempts == 0:
        print("Maximum attempts reached. Could not get valid JSON.")
        return

    # content = agent({"input": "Top 10 " + query})
    search_results = asyncio.run(search_amazon(query))

    if search_results is None:
        print("Search failed. Exiting.")
        return

    # Extract URLs from search results
    urls = [result['url'] for result in search_results]

    # Initialize an empty list to hold all the scraped data
    all_product_details = []

    # Counter for the number of products with image URLs
    product_count_with_images = 0

    # Create a dictionary for easy lookup of search result items by URL
    search_results_dict = {result['url']: result for result in search_results}

   # Step 2: Loop through each URL to scrape data.
    for url in urls:
        product_details = asyncio.run(
            scrape_website('Scrape product details', url))
        
        print(product_details)
        search_result_item = search_results_dict.get(url, {})
        price = search_result_item.get('price', 'N/A')

        if isinstance(product_details, dict):
            # Check if the product has an image URL
            snippet = product_details.get('snippet', 'N/A')
            if product_details.get('Images') and product_details['Images'][0].get('url').strip() not in ["", "N/A"]:
                product_details['Price'] = str(price)
                product_details['Description'] = snippet

                all_product_details.append(product_details)
                product_count_with_images += 1

                # Stop if we've gathered 10 products with image URLs
                if product_count_with_images >= 10:
                    break
        else:
            print("Warning: product_details is not a dictionary")

    actual_content = all_product_details

    try:
        if (is_valid_json):
            save_to_airtable(remove_duplicate_json(
                actual_content), query, unique_id, type)
        else:
            print(f"Invalid JSON received. Attempts left: {max_attempts - 1}")
            long_running_task(query, unique_id, max_attempts=max_attempts - 1)
    except Exception as e:
        print(f"An error occurred: {e}")


@app.post("/")
def researchAgentV2(query: Query):
    type = query.type
    query = query.query
    
    unique_id = str(uuid.uuid4())
    # Start a new thread for the long-running task
    thread = Thread(target=long_running_task, args=(query, unique_id, type))
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

def get_airtable_api_id(type):
    if type == "US Baby":
        return "appMIkd5mMSKDXzkr"
    elif type == "UK Baby":
        return "appPA8CS25feRpk84"
    elif type == "UK Beauty":
        return "appxmV9N7mo372EkX"
    else:
        raise ValueError("Invalid type specified")
    
    
def get_make_api_url(type):
    if type == "US Baby":
        return "https://hook.eu1.make.com/5uyqhpqm1beskwadyysebuvq23na7734"
    elif type == "UK Baby":
        return "https://hook.eu1.make.com/m198nfyf5pus5ijd4svjjyjbup9n2148"
    elif type == "UK Beauty":
        return "https://hook.eu1.make.com/zrkuo3gwed1duqykaohastd573u1jat6"
    else:
        raise ValueError("Invalid type specified")
    

def save_to_airtable(all_product_details, category, unique_id, type):
    API_URL = "https://api.airtable.com/v0/" + get_airtable_api_id(type) + "/Products"

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
        API_URL = "https://api.airtable.com/v0/" + get_airtable_api_id(type) + "/Generated%20Articles"

        data = {
            "fields": {
                "batch_id": unique_id
            }}

        requests.post(API_URL, headers=headers, json=data)

        # Execute Generation of Artile Scenario in Make
        API_URL = get_make_api_url(type)

        data = {
            "batch_id": unique_id
        }

        requests.get(API_URL, json=data)

        print("Record successfully added.")
    else:
        print(f"Failed to add record: {response.content}")
