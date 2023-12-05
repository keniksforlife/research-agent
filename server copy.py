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
import http.client
from urllib.parse import quote
import openai


load_dotenv()
brwoserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")
airtable_key = os.getenv("AIRTABLE_API_KEY")
scrape_ant_key_1 = os.getenv("SCRAPING_ANT_KEY_1")
scrape_ant_key_2 = os.getenv("SCRAPING_ANT_KEY_2")
open_ai_key = os.getenv("OPENAI_API_KEY")

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
    if not text1.strip() or not text2.strip():
        # One or both texts are empty
        return 0

    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
    except ValueError:
        # Handling the case where TF-IDF fails (e.g., due to stop words)
        return 0



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

def search_amazon(query):
    product_details = []
    unique_amazon_results = []
    page_number = 1
    max_pages = 5  # Limit to prevent too many requests

    while len(unique_amazon_results) < 15 and page_number <= max_pages:
        print(f"Searching products for {query} - Page {page_number}")

        # Prepare the URL for ScrapingAnt API for each page
        encoded_query = quote(query)
        api_url = f"https://api.scrapingant.com/v2/general?url=https%3A%2F%2Fwww.amazon.com%2Fs%3Fk%3D{encoded_query}&page={page_number}&x-api-key={scrape_ant_key_1}&proxy_country=US"

        try:
            response = requests.get(api_url)
            response.raise_for_status()

            # Process the HTML content
            soup = BeautifulSoup(response.content, "html.parser")

            search_results = soup.find_all('div', {'data-component-type': 's-search-result'})
            if not search_results:
                break  # Break if no results found on the page

            for item in search_results:
                asin = item['data-asin']
                if not asin:
                    continue

                product_info = {'asin': asin}

                title_element = item.select_one('.a-size-medium.a-color-base.a-text-normal, .a-size-base-plus.a-color-base.a-text-normal, .a-size-base.a-color-base')
                product_info['title'] = title_element.get_text(strip=True) if title_element else "N/A"

                # Extract URL
                url_element = item.select_one('.a-link-normal.s-no-outline')
                product_info['url'] = 'https://www.amazon.com' + url_element['href'] if url_element else None

                # Extract Price
                price_element = item.select_one('span.a-price > span.a-offscreen')
                product_info['price'] = price_element.get_text(strip=True) if price_element else "N/A"

                # Extract Rating
                rating_element = item.select_one('.a-icon-star-small')
                if rating_element:
                    rating_text = rating_element.get_text(strip=True)
                    product_info['rating'] = rating_text.split(' ')[0]

                # Extract Number of Reviews
                reviews_element = item.select_one('.a-size-small .a-size-base')
                reviews_count =  reviews_element.get_text(strip=True) if reviews_element else "N/A"

                try:
                    # Remove commas and convert to integer
                    product_info['reviews_count'] = int(reviews_count.replace(',', '')) if reviews_count.replace(',', '').isdigit() else 0
                except ValueError:
                    # Handle cases where conversion to integer fails
                    product_info['reviews_count'] = 0

                # Extract Image
                image_element = item.select_one('img.s-image')
                product_info['image'] = image_element['src'] if image_element else None

                product_details.append(product_info)


            #End For search_results
            product_details.extend([product for product in product_details if product['asin'] not in [p['asin'] for p in unique_amazon_results]])
            unique_amazon_results = filter_similar_products(product_details)

            print(unique_amazon_results)
            page_number += 1

        except requests.RequestException as e:
            print(f"Error during requests to ScrapingAnt API: {e}")
            break

    print(f"After filtering, {len(unique_amazon_results)} unique Amazon results remain.")
    sorted_product_details = sorted(unique_amazon_results, key=lambda x: x['reviews_count'], reverse=True)
    return sorted_product_details


# print(search_amazon("robot vacuum cleaners"))
# 2. Tool for scraping
# 

async def is_product_image(image_url):

    print(f'checking if the {image_url} is a lifestyle image or not')

    image_analysis_prompt = """
      Please analyze the image carefully. Consider the following aspects to determine whether it is more likely a lifestyle image or a product image:

      1. Context: Is the product shown in a real-life setting or scenario? Are there people interacting with it in a way that suggests everyday use or a certain lifestyle?
      2. Focus: Is the primary focus on the product itself, with clear, detailed views of its features, or is the product part of a larger scene or narrative?
      3. Emotional Appeal: Does the image seem to be telling a story or creating an emotional connection, suggesting how the product enhances a particular lifestyle or experience?
      4. Background: Is the product displayed against a plain and neutral background or within a setting that adds context to its use?

      Based on these criteria, is this image a lifestyle image (Yes/No)? Please only answer with Yes or No.
      """

    # Craft the prompt for GPT
    prompt_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": image_analysis_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }
            ]
        }
    ]

    # Send a request to GPT
    params = {
        "model": "gpt-4-vision-preview",
        "messages": prompt_messages,
        "api_key": open_ai_key,
        "headers": {"Openai-Version": "2020-11-07"},
        "max_tokens": 1000,
    }

    result = openai.ChatCompletion.create(**params)
    response = result.choices[0].message.content.strip().lower()

    # Check if the response contains 'yes' or 'no' and return accordingly
    if 'yes' in response:
        print('lifestyle image')
        return False
    elif 'no' in response:
        print('product image')
        return True
    else:
        print("Response does not contain a clear Yes or No.")
        return None

async def scrape_website_ant(objective: str, url: str):
    try:
        print(f"Start Scraping {url}")

        # Prepare the URL for ScrapingAnt API
        encoded_url = quote(url)
        api_url = f"https://api.scrapingant.com/v2/general?url={encoded_url}&x-api-key={scrape_ant_key_2}&proxy_country=US"

        response = requests.get(api_url)
        response.raise_for_status()

        # Process the HTML content
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract the required data using BeautifulSoup
        name_elem = soup.select_one('span.product-title-word-break')
        details_elem = soup.select_one('div#detailBullets_feature_div')
        details_elem2 = soup.select_one('div#productDetails_feature_div')
        details_elem3 = soup.select_one('div#prodDetails')
        description_elem = soup.select_one('#productDescription p, #productDescription')



        #Image URLs Logic

        image_url = "N/A"
       # Attempt to fetch high-resolution images using regex
        images = re.findall('"hiRes":"([^"]+)"', response.text)
        all_images = images[:5]  # Limit to first 5 images
        
        if all_images:
            print(all_images)
            # Check each image to find a product image
            for img_url in all_images:
                if await is_product_image(img_url):
                    image_url = img_url
                    break
            
        
        if image_url == "N/A" and not all_images:
            image_url_elem = soup.select_one('img[data-old-hires]')
            image_url_elem2 = soup.select_one('[data-action="main-image-click"] img')
            image_url = image_url_elem['data-old-hires'] if image_url_elem else "N/A"

            if image_url == "N/A":
                image_url = clean_text(str(image_url_elem2.text.strip())) if image_url_elem2 else "N/A"
            

        #End Image URLs Logic


        if not description_elem:
            description_elem = soup.select_one('.a-expander-content')

        about_elem = soup.select_one('#feature-bullets ul, #feature-bullets')

        # Check for None before accessing attributes
        name = str(name_elem.text.strip()) if name_elem else "N/A"

        details = clean_text(str(details_elem.text.strip())
                                ) if details_elem else "N/A"
        
        # Check for None before accessing attributes
        description = description_elem.get_text(strip=True) if description_elem else "N/A"
        about = about_elem.get_text(strip=True) if about_elem else "N/A"

        about = str(about_elem.text.strip()) if about_elem else "N/A"
    
        if (details == "N/A"):
            details = clean_text(
                str(details_elem2.text.strip())) if details_elem2 else "N/A"

        if (details == "N/A"):
            details = clean_text(
                str(details_elem3.text.strip())) if details_elem3 else "N/A"


        # Construct and return the product details dictionary
        product_details = {
            "sp_name": name,
            "Images": [
                {
                    "url": is_valid_url(image_url)
                }
            ],
            "Image Link": is_valid_url(image_url),
            "sp_other_details": details,
            "Description": description,
            "sp_description": description,
            "sp_about": about,
            "Buy Link": url,
        }

        print(product_details)

        if (name == "N/A"):
            return "Not a valid product content. Please find another product."
        else:
            return product_details


    except requests.RequestException as e:
        print(f"Error during requests to ScrapingAnt API: {e}")
        return "Network error occurred."
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return "An unexpected error occurred."
# asyncio.run(scrape_website_ant("","https://www.amazon.com/Adjustable-Memory-Pillow-Sleepers-Bamboo/dp/B0172GSQ7S"))

async def scrape_amazon_critical_reviews(asin):
    try:
        print(f"Start Scraping Critical Reviews for ASIN: {asin}")

        # Construct the URL for the critical reviews page
        reviews_url = f"https://www.amazon.com/product-reviews/{asin}/ref=cm_cr_arp_d_viewopt_sr?filterByStar=critical&pageNumber=1"
        encoded_url = quote(reviews_url)
        api_url = f"https://api.scrapingant.com/v2/general?url={encoded_url}&x-api-key={scrape_ant_key_1}&proxy_country=US"

        response = requests.get(api_url)
        response.raise_for_status()

        # Process the HTML content
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract review elements
        reviews = []
        review_elements = soup.select("#cm_cr-review_list .review")
        for review_elem in review_elements[:5]:
            # Extract the review text
            review_text_elem = review_elem.select_one('span[data-hook="review-body"]')
            review_text = ' '.join(text for text in review_text_elem.stripped_strings) if review_text_elem else "N/A"

            # Extract the review title
            review_title_elem = review_elem.select_one('*[data-hook="review-title"] > span')
            review_title = review_title_elem.get_text(strip=True) if review_title_elem else "N/A"

            # Extract the location and date
            location_and_date_elem = review_elem.select_one('span[data-hook="review-date"]')
            location_and_date = location_and_date_elem.get_text(strip=True) if location_and_date_elem else "N/A"

            # Check if the review is from a verified purchase
            verified_elem = review_elem.select_one('span[data-hook="avp-badge"]')
            verified = bool(verified_elem.get_text(strip=True)) if verified_elem else False

            # Extract the rating
            rating_elem = review_elem.select_one('*[data-hook*="review-star-rating"]')
            rating = rating_elem.get_text(strip=True) if rating_elem else "N/A"
            
            reviews.append({
                'title': review_title,
                'text': review_text,
                'location_and_date': location_and_date,
                'verified': verified,
                'rating': rating,
            })

        print(f"Found {len(reviews)} critical reviews for ASIN: {asin}")
        return reviews

    except requests.RequestException as e:
        print(f"Error during requests to ScrapingAnt API: {e}")
        return "Network error occurred."
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return "An unexpected error occurred."

# asyncio.run(scrape_amazon_critical_reviews("B08DY55SQZ"))

def format_reviews_for_airtable(reviews):
    # Concatenate the text of each review, separated by a specific delimiter
    delimiter = " | "  # You can change this to any delimiter you prefer
    formatted_reviews = delimiter.join(review['text'] for review in reviews)

    # Optionally truncate the string to fit within Airtable's character limit
    max_length = 10000  # Example limit, adjust based on Airtable's actual limit
    if len(formatted_reviews) > max_length:
        formatted_reviews = formatted_reviews[:max_length] + "..."

    return formatted_reviews


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
    search_results = search_amazon(query)

    print("Search Results", search_results)
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
            scrape_website_ant('Scrape product details', url))
        
        search_result_item = search_results_dict.get(url, {})
        price = search_result_item.get('price', 'N/A')
        asin = search_result_item.get('asin')
        review_count = search_result_item.get('reviews_count')
        rating = search_result_item.get('rating')

        if isinstance(product_details, dict):

            if price == 'N/A':
                print(f"Invalid or no price for product at {url}, skipping.")
                continue

            if asin:
                product_reviews = asyncio.run(scrape_amazon_critical_reviews(asin))
                formatted_reviews = format_reviews_for_airtable(product_reviews)
                
                product_details['Reviews'] = formatted_reviews
                product_details['Review Count'] = review_count
                product_details['Rating'] = rating

            if product_details.get('Images') and product_details['Images'][0].get('url').strip() not in ["", "N/A"]:
                product_details['Price'] = str(price)
                # product_details['Description'] = snippet

                all_product_details.append(product_details)
                product_count_with_images += 1

                # Stop if we've gathered 10 products with image URLs
                if product_count_with_images >= 15:
                    break
        else:
            print("Warning: product_details is not a dictionary")

    actual_content = all_product_details

    print(actual_content)
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
def researchAgent(query: Query):
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
    

def save_to_airtable1(all_product_details, category, unique_id, type):
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

def save_to_airtable(all_product_details, category, unique_id, type):
    API_URL = "https://api.airtable.com/v0/" + get_airtable_api_id(type) + "/Products"

    print("SAVING RECORDS TO AIRTABLE . . .")
    headers = {
        "Authorization": f"Bearer {airtable_key}",
        "Content-Type": "application/json"
    }

    # Function to send data to Airtable
    def send_data_to_airtable(data):
        response = requests.post(API_URL, headers=headers, json={"records": data})
        if response.status_code != 200:
            print(f"Failed to add record: {response.content}")
            return False
        return True

    # Splitting data into batches of 10 records each
    for i in range(0, len(all_product_details), 10):
        batch = all_product_details[i:i + 10]
        data_list = [{"fields": item} for item in batch]
        if not send_data_to_airtable(data_list):
            print("Error in saving batch to Airtable.")
            break

    API_URL = "https://api.airtable.com/v0/" + get_airtable_api_id(type) + "/Generated%20Articles"
    data = {"fields": {"batch_id": unique_id}}
    requests.post(API_URL, headers=headers, json=data)

    API_URL = get_make_api_url(type)
    requests.get(API_URL, params={"batch_id": unique_id})

    print("Record successfully added.")


all_product_details = [
  {
    "sp_name": "KidKraft Ultimate Corner Wooden Play Kitchen with Lights & Sounds, Play Phone and Curtains, Espresso, Gift for Ages 3+",
    "Images": [
      {
        "url": "https://m.media-amazon.com/images/I/71Lh7WuyHkL._AC_SL1500_.jpg"
      }
    ],
    "Image Link": "https://m.media-amazon.com/images/I/71Lh7WuyHkL._AC_SL1500_.jpg",
    "sp_other_details": "Product information Product Dimensions 42.5 x 32.5 x 36.75 inches Item Weight 3.52 ounces Country of Origin China ASIN B01C49MCCS Item model number 53365 Manufacturer recommended age 36 months - 10 years Batteries 9 LR44 batteries required. (included) Best Sellers Rank #1,420 in Toys & Games (See Top 100 in Toys & Games) #7 in Toy Kitchen Sets #11 in Preschool Kitchen Sets & Play Food Customer Reviews 4.7 4.7 out of 5 stars 17,053 ratings 4.7 out of 5 stars Is Discontinued By Manufacturer No Release date September 9, 2019 Manufacturer KidKraft Warranty & Support Product Warranty: For warranty information about this product, please click here. [PDF ] Feedback Would you like to tell us about a lower price? KidKraft Ultimate Corner Wooden Play Kitchen with Lights & Sounds, Play Phone and Curtains, Espresso, Gift for Ages 3+ Share: Found a lower price? Let us know. Although we can't match every price reported, we'll use your feedback to ensure that our prices remain competitive. Where did you see a lower price? Website (Online) URL: Price: ($) Shipping cost: ($) Date of the price: 01 02 03 04 05 06 07 08 09 10 11 12 / 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 / Store (Offline) Store name: City: State: Please select province Please select province Price: ($) Date of the price: 01 02 03 04 05 06 07 08 09 10 11 12 / 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 / Please sign in to provide feedback. Submit Feedback",
    "Description": "Our KidKraft Ultimate Corner Play Kitchen Set with Lights and Sounds is loaded with fun interactive features to engage your little chefs and keep them cooking. The stovetop burners and ice maker both feature lights and sounds, oven knobs turn and click and appliance doors really open and close. Functional shelving and cabinets with doors that open and close provide plenty of convenient storage for any KidKraft kitchen accessories. With so much to explore, kids will feel like they're cooking just like Mom and Dad!",
    "sp_description": "Our KidKraft Ultimate Corner Play Kitchen Set with Lights and Sounds is loaded with fun interactive features to engage your little chefs and keep them cooking. The stovetop burners and ice maker both feature lights and sounds, oven knobs turn and click and appliance doors really open and close. Functional shelving and cabinets with doors that open and close provide plenty of convenient storage for any KidKraft kitchen accessories. With so much to explore, kids will feel like they're cooking just like Mom and Dad!",
    "sp_about": "About this item    MADE OF WOOD & EASY ASSEMBLY: Kitchen is made of premium, sustainable materials for long-lasting play you can feel good about. Make assembly easier with more help! Two people can set up this item in approximately 2 hours or less.    FITS IN YOUR SPACE: Innovative curved design cozies up in the corner, letting you utilize every inch oof your room.    LIGHTS & SOUNDS: The burners light up and make cooking sounds; water dispenser also lights up and makes realistic sounds when pressed.    LAUNDRY TIME: Built-in washer with open and close inset circle window door adds cleaning to the kitchen area.    SO DETAILED: Kids will find plenty of hands-on learning with buttons to press, knobs to turn and handles to pull. And, with homey touches like fabric curtains, a wall phone and lots of storage, this wooden kitchen is almost just like the real thing!",
    "Buy Link": "https://www.amazon.com/KidKraft-Ultimate-Corner-Kitchen-Lights/dp/B01C49MCCS/ref=sr_1_1?keywords=Best+Play+Kitchen+for+Kids&qid=1701765527&sr=8-1",
    "Reviews": "Ok, so the score might be a LITTLE low for what this kitchen really is, but that's because I had really high hopes for it, based on it's look, it's features, and most importantly, it's price. Granted, $250 isn't a lot in the realm of play kitchens from other places...) but it's still a lot for a CHILDREN'S TOY. That said, this play kitchen has some good, and some not so good. Good: - It's beautiful. It looks like a real kitchen; and so much so, that I basically removed all of my daughters cheap, cartoony, brightly colored accessories. I only want her to have things that look as nice as this kitchen. - It's pretty sturdy. It's wood, and held together with screws. It's got a decent heft to it, and looks like it can take some weight and jostling. My daughter tried to climb through the drapes part, and it didn't even wobble. - assembly was pretty straight forward, with one glaring exception, which we'll get to... - Customer service was helpful, when I needed to reach out to them....which we'll get to... Not so good: - it's quite a bit smaller than I had expected, even with knowing the dimensions beforehand. For example, the \"ice maker\" accessory can only fit like the TINIEST of cups in there. (Think Dixie Cup). Anything bigger, and you can't use the ice maker. - A few pieces came damaged in shipping. Plastic parts were broken off of quite a few of the accessories, and customer service, while helpful, required proof of purchase and proof of damage before they would send me anything. - There are a LOT of parts and pieces, but they are clearly labeled and the instructions are easy to follow, BUT, the hardware is absolutely ridiculous. For example, the screws are labeled by letter. eg \"L\", but there are TWO \"L\" screws; one with painted white tops, and one without. The instructions make no mention of which screw to use and when; it just says \"L.\" So, I didn't even notice until I ran out of \"L\" screws, and the instructions asked for another, that I had to go back and look, and realized that there are 2 different L screws. Again, they both JUST say \"L,\" and the instructions JUST say \"L.\" You have to figure out on your own when to use the painted screws, and when not to. Super frustrating, because if you notice too late, like I did, you either have to just carry on, or take apart the whole thing to replace the screws. - Although it has sights and sounds, I'm actually MOST disappointed with those. The burners light up, but only for about 10 seconds at a time, and they play an incredibly weird sound, which is supposed to be the gas burner lighting, (the \"tick, tick, tick, whoosh\" sound), and then boiling water. The boiling water sound sounds like an alien language, or some computer modem wailing. They only make this noise when something is placed on them and presses the button on top, and then they cut off after their 10 second programming. To hear the sound again, you have to push the button again, at which point they play the same sound. I would've thought they would stay on until you turned them off, and might play a \"sizzling\" sound, or even a crackling fire sound, but nope.....whoosh and alien babble is what you get. Also, the ice maker only makes a noise when you push the lever, even though it has buttons that depress. The sound the ice maker makes sounds like a bunch of tin cans rattling around, which is their interpretation of ice falling into a glass. The microwave makes no sounds, and has painted on, wooden buttons. It's weird, because they include an egg timer that dings, with the kitchen; why not incorporate that into the microwave?! I had to manually add motion lights in the oven, fridge and microwave, since there are no lights on the kitchen, other than the burners. The sights and sounds on this kitchen are a complete bust, and it was one of the reasons I went with this kitchen over others. I'm not upset with my purchase, but I did bypass larger, more elaborate kitchens (also, slightly cheaper), because this one had the sights and sounds that I THOUGHT would make this one more realistic. I would ALMOST rather have no sounds, than the ones that came with this kitchen. If I had to do it all again, I would have gone with something else, but it's not like I feel totally unhappy with this purchase. My daughter seems to like it. (She is added for height reference in the pictures. She is an average sized 3 year old.) | My husband said it was reallyyyy hard to put together.. my son likes it, though it could be way more interactive especially the microwave. I don’t like the back of it. I have a very open house and the back of the kitchen is visible and it’s not the prettiest. I wish it would have been plain brown wood on the back. All in all it seems like good quality and I hope my son likes it for the long haul. I did buy it at a high price point, so people getting it on sale, it will definitely be worth it. But buying it at a higher price, it kind of falls short of that higher expectation | Have you stumbled across this kitchen set while searching for the perfect gift for your kiddo? Are you telling yourself that little Timmy would love waking up to this and finding it assembled under the tree? Well then, let you enlighten you on this kitchen set from the darkest corners of Hell. Let’s start with the basics, it looks great and it’s sturdy, great quality fake wood. I also love the sound effects and the lights, also a plus. But now let’s talk about how the devil himself constructed this kitchen set because he wanted to watch the world burn. For starters, it says no tools required-wrong. You need a screw driver. But not every hole is pre-drilled, so honestly an electric screwdriver (or drill) is your best friend for this. Speaking of holes, ya know when you’re making hanky-panky with your love partner and the holes sometimes don’t line up? That’s basically how this kitchen set works. Not every hole lines up with its partner hole, so you have to slide, move, and stretch some things to make it work. Now speaking of screwing, they include the screws needed, or not. It’s a hit or miss if they actually provided you with enough of one size. And they don’t specify what screw should go where. For example, there are 3 different type of “I” screw, but it never says what type, just “I”. It also called for the “N” screw 4 or 6 times but only gave us 2. Assembly is okay, it’s definitely a two person job and once again, there are easier ways to put things where they go if you did it in another order. But you won’t learn that until you’re almost done. Also, you’re probably going to put things in upside down and backwards, that’s expected because once again, the instructions aren’t that clear. Time it takes to complete? It took me 3 hours (not counting the hour break I took to look for a wine opener). So yes, this is a cute kitchen and it’s built well. But assembly sucks and I would honestly rather pay the price to find a used on Marketplace or a buy/sell page rather than try to assemble this thing again. In fact, I would rather give birth Again to my 10lb daughter without drugs, rather than assemble another one of these kitchens. But, I’m sure there are positive reviews out there and y’all will just assume I was upset or mad when I made this review. But that’s the exact opposite. I’m writing this review in the bathtub while drinking wine, so I have nothing to be upset about. Hope you make a wise choice while purchasing a kitchen. | So my daughter loves her kitchen.  I have a few issues with it.  Here's my review. Pros... Looks sleek and nice.  Seems like sturdy wood but we shall see over time how well it holds up. I liked that all the parts were labeled with a sticker so I didn't need to use pictures to try and figure that out.  All holes were drilled correctly and 99% of screws went right in with no issues and lined up perfectly.  All parts were included and they even had a pile of extra misc fasteners included. The stove bubbles and lights up.  The knobs click.  The ice maker lights up and makes sound.  My daughter loves it. Cons... Sooo many parts.  I'm used to putting ikea furniture together and this was WAY  more effort.  There were 49 labeled wood pieces plus the plastic pieces and like 150 screws. The screws were nicely labeled by size in a bubble package ... until I picked it up and they all fell out because the plastic bubbles were cracked.  Also,  some are painted to go on the outside so they  blend in,  but as you build it you need to guess which will be visible.  It doesn't tell you which to use. So annoying.  The directions were fairly clear but no words.  Some pictures took a few minutes to figure out how to orient pieces based on tiny screw holes. My wrist was cramped and killing me when I finished it from all the screws. I don't recommend a power driver but at least get yourself a ratcheting one (which o did use).  I did complete it myself without a helper and it took just over 3 hours. Some of the screws were in hard to reach places so make sure you have a short screw driver on hand to squeeze in tiny places.  There was one on the sink/counter that I just left out because I have no idea how you were supposed to manipulate it to get in the awkward angle.  I  was disappointed that the microwave and phone only have a painted piece.  My daughter wanted to push the buttons after the cool stove and ice maker features and there aren't any.  I thought the sink could make noise or something too. Why is there a washing machine but no dishwasher?? | I’m no stranger to building projects like this and since having my daughter I’ve built everything from dresser cubes to cribs and all the toys in between. This was by far the most frustrating project I’ve ever done. The way the parts are set up made it nearly impossible to easily screw in a few of the screws. Even with a drill and blistering my hand, a few wouldn’t go in all the way. (Yes they were correct I checked..4 times) I started building this around 7:30 pm. Just got finished...",
    "Review Count": 17053,
    "Rating": "4.7",
    "Price": "$191.24"
  },
  {
    "sp_name": "Step2 Best Chefs Kids Kitchen Play Set, Interactive Play with Lights and Sounds, Toddlers 2-5 Years Old, Realistic 25 Piece Toy Accessory Set",
    "Images": [
      {
        "url": "https://m.media-amazon.com/images/I/81EjGiaQfcL._AC_SL1500_.jpg"
      }
    ],
    "Image Link": "https://m.media-amazon.com/images/I/81EjGiaQfcL._AC_SL1500_.jpg",
    "sp_other_details": "Product information Product Dimensions 11.5 x 34.38 x 38.5 inches Item Weight 1.98 pounds Country of Origin USA ASIN B00XPH0618 Item model number 85319 Manufacturer recommended age 24 months - 5 years Batteries 2 AAA batteries required. Best Sellers Rank #1,361 in Toys & Games (See Top 100 in Toys & Games) #2 in Toddler Kitchen Play #6 in Toy Kitchen Sets #10 in Preschool Kitchen Sets & Play Food Customer Reviews 4.7 4.7 out of 5 stars 12,428 ratings 4.7 out of 5 stars Is Discontinued By Manufacturer No Release date June 1, 2018 Manufacturer Step2 Warranty & Support Product Warranty: For warranty information about this product, please click here. [PDF ] Feedback Would you like to tell us about a lower price? Step2 Best Chefs Kids Kitchen Play Set, Interactive Play with Lights and Sounds, Toddlers 2-5 Years Old, Realistic 25 Piece Toy Accessory Set Share: Found a lower price? Let us know. Although we can't match every price reported, we'll use your feedback to ensure that our prices remain competitive. Where did you see a lower price? Website (Online) URL: Price: ($) Shipping cost: ($) Date of the price: 01 02 03 04 05 06 07 08 09 10 11 12 / 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 / Store (Offline) Store name: City: State: Please select province Please select province Price: ($) Date of the price: 01 02 03 04 05 06 07 08 09 10 11 12 / 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 / Please sign in to provide feedback. Submit Feedback",
    "Description": "Our amazing Learn & Play Kitchen Set! This playhouse is intended to encourage imagination and creativity with its realistic molded-in sink, faucet, huge countertop, refrigerator, and oven with functional doors. It promotes role-play, contributes to the development of social skills, and assists youngsters in making sense of real-life events. The interactive 25-piece accessory kit included with the kitchen set increases fun by giving pots, pans, utensils, a coffee pot, and more. Please be aware that food models are not included. Our Learn & Play Kitchen Set is perfect for saving space. Its compact design allows it to easily fit into small areas. Weighing just 22.5 pounds. This lightweight playset, measuring 35.8\" H x 34.4\" W x 11.5\" D, is easy to carry around, allowing your child to play anywhere they like. Effortlessly organize and pack all of the kitchen's equipment, maintaining a clutter-free environment and making cleanup effortless for parents. We take pleasure in making high-quality items, and the Learn & Play Kitchen Set is no exception. This playground is made of robust double-walled plastic and is built to last. The vivid colors are resistant to chipping, fading, cracking, and peeling, ensuring years of usage and delight.",
    "sp_description": "Our amazing Learn & Play Kitchen Set! This playhouse is intended to encourage imagination and creativity with its realistic molded-in sink, faucet, huge countertop, refrigerator, and oven with functional doors. It promotes role-play, contributes to the development of social skills, and assists youngsters in making sense of real-life events. The interactive 25-piece accessory kit included with the kitchen set increases fun by giving pots, pans, utensils, a coffee pot, and more. Please be aware that food models are not included. Our Learn & Play Kitchen Set is perfect for saving space. Its compact design allows it to easily fit into small areas. Weighing just 22.5 pounds. This lightweight playset, measuring 35.8\" H x 34.4\" W x 11.5\" D, is easy to carry around, allowing your child to play anywhere they like. Effortlessly organize and pack all of the kitchen's equipment, maintaining a clutter-free environment and making cleanup effortless for parents. We take pleasure in making high-quality items, and the Learn & Play Kitchen Set is no exception. This playground is made of robust double-walled plastic and is built to last. The vivid colors are resistant to chipping, fading, cracking, and peeling, ensuring years of usage and delight.",
    "sp_about": "About this item    LEARN & PLAY: Entertain your kid for hours with realistic molded-in sink with faucet, large countertop, refrigerator, and oven with working doors, supports role play, develop social skills, make sense of real-life situations    INTERACTIVE TOYS: Realistic lights and sounds, 25-piece kitchen set, enhance playtime with pots, pans, silverware, coffee pot, and more, food models and batteries not included    SPACE SAVER: Compact, fits into small spaces, lightweight, easy to assemble, move around and clean, weighs 22.5 lbs., dimensions: 35.8” H x 34.4” W x 11.5” D    EASY STORAGE: When playtime is over, let your toddler tidy up with shelves, hooks, and a recycling bin to stow away their toys    DURABLE: Built to last and easy to clean, durable double-walled plastic construction, years of use with colors that won't chip, fade, crack, or peel",
    "Buy Link": "https://www.amazon.com/Step2-Best-Chefs-Kitchen-Playset/dp/B00XPH0618/ref=sr_1_33?keywords=Best+Play+Kitchen+for+Kids&qid=1701765540&sr=8-33",
    "Reviews": "It was pretty disappointing setting this up the night before Christmas for my daughter to find out it didn't come with ANY of the stickers for the kitchen or the condiments. | The finished product is great. But, lawd help me trying to put it together was a task. Requires you to screw in the large plastic parts without predrilled holes. After that fiasco, it comes with the stickers for YOU to apply without any direction as to where they go. Be prepared. It’s a doozy. | Stove stopped working after day 1 | The media could not be loaded. The product is nice and sturdy but there are gouges and scratches  all along the front of the kitchen set.  One of the screws for the burner were stripped so i cannot get the burner screwed down.  The gouges on the front make this look like a used product which is very upsetting being I bought it for my granddaughters 2nd birthday present. | The media could not be loaded. This is made well. My Grandson loves it! I’m just disappointed that is was smaller than I expected. Otherwise nice",
    "Review Count": 12428,
    "Rating": "4.7",
    "Price": "$89.99"
  },
  {
    "sp_name": "Best Choice Products Pretend Play Kitchen Wooden Toy Set for Kids w/Realistic Design, Telephone, Utensils, Oven, Microwave, Sink - White",
    "Images": [
      {
        "url": "https://m.media-amazon.com/images/I/71uOEvw9gtL._AC_SL1500_.jpg"
      }
    ],
    "Image Link": "https://m.media-amazon.com/images/I/71uOEvw9gtL._AC_SL1500_.jpg",
    "sp_other_details": "Product information Product Dimensions 31.25 x 9.5 x 31.5 inches Item Weight 28.5 pounds Country of Origin China ASIN B08JQQZ7R2 Manufacturer recommended age 3 years and up Best Sellers Rank #1,098 in Toys & Games (See Top 100 in Toys & Games) #4 in Toy Kitchen Sets #8 in Preschool Kitchen Sets & Play Food Customer Reviews 4.3 4.3 out of 5 stars 5,183 ratings 4.3 out of 5 stars Manufacturer Best Choice Products Warranty & Support Product Warranty: For warranty information about this product, please click here Feedback Would you like to tell us about a lower price? Best Choice Products Pretend Play Kitchen Wooden Toy Set for Kids w/Realistic Design, Telephone, Utensils, Oven, Microwave, Sink - White Share: Found a lower price? Let us know. Although we can't match every price reported, we'll use your feedback to ensure that our prices remain competitive. Where did you see a lower price? Website (Online) URL: Price: ($) Shipping cost: ($) Date of the price: 01 02 03 04 05 06 07 08 09 10 11 12 / 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 / Store (Offline) Store name: City: State: Please select province Please select province Price: ($) Date of the price: 01 02 03 04 05 06 07 08 09 10 11 12 / 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 / Please sign in to provide feedback. Submit Feedback",
    "Description": "Go to Your Orders to start the returnPrint the return shipping labelShip it!",
    "sp_description": "Go to Your Orders to start the returnPrint the return shipping labelShip it!",
    "sp_about": "About this item    CHALKBOARD SURFACE: Little chefs can jot down notes on the included chalkboard surface as they play grownup all day with this large set made with a realistic design that mimics a real kitchen, designed for their size, and have fun playing with the ice maker, microwave, phone, and other fun features    11 ACCESSORIES INCLUDED: Set includes 3 utensils, 2 pots, one lid to play with and kitchen comes with an ice maker that dispenses ice cubes (4), and even a cordless phone; keep the accessories in the 20.25 sq. in overhead shelf, the 110 sq. in. oven, or in the fridge, freezer, microwave, and dishwasher    TRUE TO LIFE KITCHEN: Your child can get a true experience with fun features like a towel rack and ice machine, and realistic sounds like the \"click\" of the oven knobs as they pretend to bake and cook in a stylishly designed kitchen with a backsplash for a modern appeal    ASSEMBLY INFORMATION: Great toys take time to build! This item has many screws that make it sturdy and resilient for kids to play with; set up your pretend kitchen by closely following the included instruction manual in an estimated assembly time of 90 min. Use a power drill to save more time!    CERTIFIED & SAFE: Rest at ease while your children play with this kitchen set that is both well-built and fun; made with plastic materials and meets U.S. Federal safety standards for ASTM & CPSIA; OVERALL DIMENSIONS: 31.25\"(L) x 9.5\"(W) x 31.5\"(H)",
    "Buy Link": "https://www.amazon.com/Best-Choice-Products-Realistic-Telephone/dp/B08JQQZ7R2/ref=sr_1_27?keywords=Best+Play+Kitchen+for+Kids&qid=1701765540&sr=8-27",
    "Reviews": "This is a nice kitchen but I am terribly disappointed that the H brackets wholes are sized to small so the doors do not fit. This kitchen also takes a couple hours to put together but other than that my granddaughter loves it. I wish for the money I sent that she could have the whole experience. | Took me hours to put together | Still haven’t had the opportunity to assemble as none of the pieces have letters. Trying to guess which is which is hard. Seems durable but still not assembled. Questioned on whether i should of gotten with a hard plastic one instead. | All my daughter asked for this Christmas was a Kitchen, so we bought her this one. It took me 3 hours to assemble the whole thing by myself. It is a really nice Kitchen, very sturdy, lots of closed door's storage and the perfect height. Even though it took me three hours there were some nice points to the assembly. I really liked how they gave you instructions to assemble all the hardware first, before actually putting the kitchen together. A few things that could have made that better was, they wanted you to install all the hardware first but when you got to assembly, there were still a few pieces that they wanted you to put on that was actually a pain to do while it was together and could have been done before the assembly. Also, the holes for the small wooden dowels that help hold it together were too big, so they would just get lost in the wood when trying to stick them to the other piece. Also, the instructions didn't tell you which holes to the put the dowels in, some pieces had 4 holes and you didn't know if you were supposed to put them on the outside holes or inside holes, most of the time you could look closely as the picture and tell, but it would have been nice if it was written out. Other than those little things, setting it up was quite easy. As far as my daughter's negative points, the sink knobs don't turn, yes, I know it's all pretend, but that would be a great thing if they were able to turn, even just a little. | This kitchen set is okay for the price. It took me about 2.5hrs to build for one person. But fairly easy because everything is labeled. I did have trouble with piece number 6 and had to take everything apart and then turn it around. You prep every number first before assemble and then prep again for the final touches. I was expecting a square faucet like in the photo but got a round one. I really wanted the square!! It doesn’t  one with chalk for the board either. In the end I had two wooden pieces left, not sure what it’s sure it’s not labeled anywhere in the book.",
    "Review Count": 5183,
    "Rating": "4.3",
    "Price": "$99.99"
  },
  {
    "sp_name": "JOYIN Kid Play Kitchen, Pretend Daycare Toy Sets, Kids Cooking Supplies with Stainless Steel Cookware Pots and Pans Set, Cooking Utensils, Apron&Chef Hat and Grocery Play Food Sets, Toddler Gifts",
    "Images": [
      {
        "url": "https://m.media-amazon.com/images/I/81nL29aQYcL._AC_SL1500_.jpg"
      }
    ],
    "Image Link": "https://m.media-amazon.com/images/I/81nL29aQYcL._AC_SL1500_.jpg",
    "sp_other_details": "Product information Product Dimensions 9.8 x 7.6 x 5.31 inches Item Weight 1.81 pounds Country of Origin China ASIN B07KPN4V2G Item model number 10724 Manufacturer recommended age 3 years and up Best Sellers Rank #73,333 in Toys & Games (See Top 100 in Toys & Games) #425 in Toy Kitchen Sets #665 in Preschool Kitchen Sets & Play Food Customer Reviews 4.7 4.7 out of 5 stars 4,674 ratings 4.7 out of 5 stars Is Discontinued By Manufacturer No Manufacturer Joyin Inc Warranty & Support Product Warranty: For warranty information about this product, please click here Feedback Would you like to tell us about a lower price? JOYIN Kid Play Kitchen, Pretend Daycare Toy Sets, Kids Cooking Supplies with Stainless Steel Cookware Pots and Pans Set, Cooking Utensils, Apron&Chef Hat and Grocery Play Food Sets, Toddler Gifts Share: Found a lower price? Let us know. Although we can't match every price reported, we'll use your feedback to ensure that our prices remain competitive. Where did you see a lower price? Website (Online) URL: Price: ($) Shipping cost: ($) Date of the price: 01 02 03 04 05 06 07 08 09 10 11 12 / 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 / Store (Offline) Store name: City: State: Please select province Please select province Price: ($) Date of the price: 01 02 03 04 05 06 07 08 09 10 11 12 / 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 / Please sign in to provide feedback. Submit Feedback",
    "Description": "Go to your orders and start the returnSelect the return methodShip it!",
    "sp_description": "Go to your orders and start the returnSelect the return methodShip it!",
    "sp_about": "MULTIPLE FEATURES: Toy Kitchen Cooking Set Bundle Includes Stainless Steel Pan and Pots with Lid and Handles, 2 Stainless Steel and 2 Wooden Utensils, Apron, Oven Mitt, Pot Holder, Chef Hat and Grocery Toys.    CHILD SAFE: Non-Toxic. Meet US toy standard.    SUPERIOR QUALITY: Not Plastic, Not Cheap Tin, Made of Real Stainless Steel! Super Durable. Easy to Use. Soft Edges. Size Perfect for Kids, Toddler and Children.    SUPER FUN AND EDUCATIONAL TOY: With this Pretend Play Kitchen Toy Set, Children can feel free to Cook with their parents, Enhance their Dexterity and Stimulate Imagination!    SUPER VALUE PACK for Girls and Boys Kids Christmas Gift. Perfect for Cooking Chef Pretend Play, Role Play, Educational Toy, , Early Development, Holiday Toy for Toddlers, School Classroom Prize, Kids Intelligent Learning Toys, Easter Stuffer, Baby Shower, Birthday party or Festival (Halloween Thanksgiving, New Year) and more.",
    "Buy Link": "https://www.amazon.com/Accessories-Stainless-Cookware-Utensils-Learning/dp/B07KPN4V2G/ref=sr_1_47?keywords=Best+Play+Kitchen+for+Kids&qid=1701765555&sr=8-47",
    "Reviews": "Faltaron piezas | Overall this set is adorable. The pots and utensils are perfect size and sturdy. My only complaint is the apron is very cheaply made. One side around the neck has Velcro which is great, the other side is sold on, but all it took was one little tug and the stitching ripped. I had to sew it back on (which I am clearly not good at on Christmas night) . The other side with the Velcro looks like it’s barely hanging on also. I think if the sewing was more sturdy I would have absolutely zero complaints about this set. That is the only reason I did not give it five stars. The food is a sturdy plastic material, my kids are not able to bite dents in them like some other brands I have purchased. | I dislike giving an item as a gift when the packaging looks bad.  Feels as though the item was used and not new. | I ordered this set for my 2-year-old toddler who had just gotten a play kitchen. The pots and pans that came with it were awesome, however the hat and pot holder were not great. The chef hat logo was cut off and on backwards and the pot holder glove would fit no living human let alone a toddler. The shape is very abnormal. | These are a great product, I just wish there was a little bit more food that can go in her pots and pans, most of it is imitation of fruit and vegetables but nothing she could pretend that he’s done I guess. But hey it’s kids toys and they are cute",
    "Review Count": 4674,
    "Rating": "4.7",
    "Price": "$17.99"
  },
  {
    "sp_name": "New Classic Toys Blue Wooden Pretend Play Toy Kitchen for Kids with Role Play Bon Appetit Electric Cooking Included Accesoires Makes Sound",
    "Images": [
      {
        "url": "https://m.media-amazon.com/images/I/61PiMluc7BL._AC_SL1500_.jpg"
      }
    ],
    "Image Link": "https://m.media-amazon.com/images/I/61PiMluc7BL._AC_SL1500_.jpg",
    "sp_other_details": "Product information Product Dimensions 23.6 x 11.8 x 39.4 inches Item Weight 21.3 pounds Country of Origin China ASIN B07C3R2N7F Item model number 11065 Manufacturer recommended age 3 years and up Best Sellers Rank #72,973 in Toys & Games (See Top 100 in Toys & Games) #422 in Toy Kitchen Sets #662 in Preschool Kitchen Sets & Play Food Customer Reviews 4.6 4.6 out of 5 stars 2,181 ratings 4.6 out of 5 stars Is Discontinued By Manufacturer No Release date November 10, 2017 Manufacturer New Classic Toys Warranty & Support Manufacturer’s warranty can be requested from customer service. Click here to make a request to customer service. Feedback Would you like to tell us about a lower price? New Classic Toys Blue Wooden Pretend Play Toy Kitchen for Kids with Role Play Bon Appetit Electric Cooking Included Accesoires Makes Sound Share: Found a lower price? Let us know. Although we can't match every price reported, we'll use your feedback to ensure that our prices remain competitive. Where did you see a lower price? Website (Online) URL: Price: ($) Shipping cost: ($) Date of the price: 01 02 03 04 05 06 07 08 09 10 11 12 / 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 / Store (Offline) Store name: City: State: Please select province Please select province Price: ($) Date of the price: 01 02 03 04 05 06 07 08 09 10 11 12 / 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 / Please sign in to provide feedback. Submit Feedback",
    "Description": "A little chef's dream kitchen from New Classic Toys for preparing and cooking playfood. Little chefs will love cooking up meals for the whole family with the adorable bon appetit kitchen. Children can use their imaginations to create menus. This kitchen is perfect for playtime as your child and their friends can cook together, promoting strong social skills. It has everything a little chef needs for hours of pretend-play fun! Complete with: • a pot with lid • frying pan • three cooking utensils • an oven glove • a towel • a salt and pepper pot This beautiful kitchen is part of the New Classic Toys \"Bon Appetit\" productline. Cutting, baking, cooking and ready for dinner! Our little pots, pans, cuttings sets and kitchens are absolute must haves for little master chefs.",
    "sp_description": "A little chef's dream kitchen from New Classic Toys for preparing and cooking playfood. Little chefs will love cooking up meals for the whole family with the adorable bon appetit kitchen. Children can use their imaginations to create menus. This kitchen is perfect for playtime as your child and their friends can cook together, promoting strong social skills. It has everything a little chef needs for hours of pretend-play fun! Complete with: • a pot with lid • frying pan • three cooking utensils • an oven glove • a towel • a salt and pepper pot This beautiful kitchen is part of the New Classic Toys \"Bon Appetit\" productline. Cutting, baking, cooking and ready for dinner! Our little pots, pans, cuttings sets and kitchens are absolute must haves for little master chefs.",
    "sp_about": "About this item    100% wood    Safety: we want to challenge, excite and enjoy children with high quality toys. All products comply with the strictest safety requirements conform the standards. Our products last for generations. And we are very proud of this.    Fun: this beautiful toy is colourful and has many fun options to play with! The color is very bright. Children would spend lot of time playing and parents finally could get some rest    Learning: Playing is not just fun, it is also very important. While playing kids get familiar with the world. Building, stacking, making music or imitating: playing contributes to the development of insight, talent and motor skills.    Age: Designed for anyone aged 3 and up, this imagination toy can be enjoyed by every child in the family    Bon Appetit: This toy set is part of the \"Bon Appetit\" collection of New Classic Toys. Cutting, baking, cooking and ready for dinner! Our little pots, pans, cuttings sets and kitchens are absolute must haves for little master chefs.",
    "Buy Link": "https://www.amazon.com/New-Classic-Toys-11065-Kitchenette/dp/B07C3R2N7F/ref=sr_1_26?keywords=Best+Play+Kitchen+for+Kids&qid=1701765527&sr=8-26",
    "Reviews": "Wife of the account holder: My husband and I were so excited to find an adorable wooden kitchen set to give our daughter for Christmas. We totally thought it was a quality product until we opened the box. Right away we noticed the wood pieces were very flimsy… not what we had expected at all. My husband began assembling and within a few seconds realized the screw holes in the first piece he picked up were not lined up at all, not level (picture above) with one another making the handle crooked after putting it on. The next piece he put together cracked ( picture above) as the screw went in. He DID NOT use a powered drill or over tightened as we already knew the wood seemed flimsy. He was very careful and slow with putting the screw in but instantly the wood cracked. The next handle he had to put on fell apart when he picked it up! The wooden pieces forming the handle were glued together but apparently not well at all. We are wondering if it’s worth assembling the rest? Is this toy a hazard for our child? We are so disappointed! This was supposed to be a gift for our daughter! I am shocked, angry and so upset that this was what we received. Here we thought we were buying a quality children’s toy that would last. It didn’t last five minutes out of the box!! Horrible! Absolutely terrible craftsmanship! Don’t waste your money!! | A birthday gift for my granddaughter | Cute but too small. Notice the dimensions. | The little kitchen is nice and cute. However, I got a wrong board. It suppose to be one LONGER board and one SHORTER boards for the two shelves, but I got two identical SHORT boards. I've halfway through the installation, so I don't want to return it. But I'm not sure how to contact the seller for a replacement of the correct boards. | Too tedious and difficult to put together! Pain. Needed 2 people",
    "Review Count": 2181,
    "Rating": "4.6",
    "Price": "$151.99"
  },
  {
    "sp_name": "Amazon Basics Kids Upright Wooden Kitchen Toy Playset with Stove, Oven, Sink, Fridge and Accessories, for Toddlers, Preschoolers, Children Age 3+ Years, White & Gray, 39\"L x 11.8\"W x 39'H'",
    "Images": [
      {
        "url": "https://m.media-amazon.com/images/I/71JQ7eY4uRL._AC_SL1500_.jpg"
      }
    ],
    "Image Link": "https://m.media-amazon.com/images/I/71JQ7eY4uRL._AC_SL1500_.jpg",
    "sp_other_details": "Product information Product Dimensions 39 x 11.8 x 39 inches Item Weight 43 pounds Country of Origin China ASIN B08P4HB3CY Item model number 838283 Manufacturer recommended age 3 years and up Best Sellers Rank #7,797 in Toys & Games (See Top 100 in Toys & Games) #51 in Toy Kitchen Sets #94 in Preschool Kitchen Sets & Play Food Customer Reviews 4.3 4.3 out of 5 stars 1,383 ratings 4.3 out of 5 stars Manufacturer Amazon Warranty & Support Manufacturer’s warranty can be requested from customer service. Click here to make a request to customer service. Feedback Would you like to tell us about a lower price? Amazon Basics Kids Upright Wooden Kitchen Toy Playset with Stove, Oven, Sink, Fridge and Accessories, for Toddlers, Preschoolers, Children Age 3+ Years, White & Gray, 39\"L x 11.8\"W x 39'H' Share: Found a lower price? Let us know. Although we can't match every price reported, we'll use your feedback to ensure that our prices remain competitive. Where did you see a lower price? Website (Online) URL: Price: ($) Shipping cost: ($) Date of the price: 01 02 03 04 05 06 07 08 09 10 11 12 / 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 / Store (Offline) Store name: City: State: Please select province Please select province Price: ($) Date of the price: 01 02 03 04 05 06 07 08 09 10 11 12 / 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 / Please sign in to provide feedback. Submit Feedback",
    "Description": "Product DescriptionAmazon Basics Kids Corner Wooden Kitchen PlaysetFrom the ManufacturerYour little chef will be up to something good in their very own kitchen with this durable and adorable kitchen playset with accessories. Perfect for toddlers and preschoolers 3 and up, they’ll learn key developmental skills through play as they practice storytelling, role play and fine motor skills. A sink, oven, stove with overhead vent, and fridge are the big pieces, with a skillet, lid, pot, ladle and spatula as accessories. The high-quality FSC-certified wood and other child-safe materials are built to last – tough play is OK with this playset. Watch your kid’s creative side blossom as they prepare elaborate feasts for their family and friends! Available in corner and upright styles.",
    "sp_description": "Product DescriptionAmazon Basics Kids Corner Wooden Kitchen PlaysetFrom the ManufacturerYour little chef will be up to something good in their very own kitchen with this durable and adorable kitchen playset with accessories. Perfect for toddlers and preschoolers 3 and up, they’ll learn key developmental skills through play as they practice storytelling, role play and fine motor skills. A sink, oven, stove with overhead vent, and fridge are the big pieces, with a skillet, lid, pot, ladle and spatula as accessories. The high-quality FSC-certified wood and other child-safe materials are built to last – tough play is OK with this playset. Watch your kid’s creative side blossom as they prepare elaborate feasts for their family and friends! Available in corner and upright styles.",
    "sp_about": "About this item    FULL TOY KITCHEN SET: This fully equipped toy kitchen set will keep kids busy for hours, day after day. The realistic playset has doors that open and close, knobs that turn and click, and sights and sounds just like a real kitchen.    ACCESSORIES INCLUDED: The wooden playset includes a toy metal spatula, ladle, pot, lid and skillet along with the main pieces: A stove with overhead vent, oven, sink and refrigerator. They’ll love cooking and serving pretend meals!    DEVELOPMENTAL TOY: Kids love imitating the adults around them, and it’s key to their early development. Through role play, they’ll build storytelling skills that are key to growing socially. They’ll also build fine motor skills and practice organization.    SAFE AND DURABLE: Made from FSC-certified wood and other child-safe materials, this deluxe toy has sturdy and lasting construction that’s safe for your children and will stand up to tough play.    FOR CHILDREN AGES 3+ YEARS: The perfect toy for curious toddlers and preschoolers learning to explore the world around them. Help them develop crucial developmental skills through play.",
    "Buy Link": "https://www.amazon.com/Amazon-Basics-Upright-Kitchen-White/dp/B08P4HB3CY/ref=sr_1_39?keywords=Best+Play+Kitchen+for+Kids&qid=1701765540&sr=8-39",
    "Reviews": "One panel was damaged when it arrived. The oven door doesn’t line up properly to catch the magnet so it doesn’t stay closed. It was easy to assemble with the directions. | We play with this DAILY, but the doors have fallen off. I do wish it was a bit more durable. | My  arrived with no screws, no kitchen utensils that were included and more than hald of the parts. I didn't realize it until I spreaded all of the parts out to get ready to put it together | The kitchen went together ok. One screw wouldn't grab properly so one of the corners isn't tight. Took about an hour to assemble by myself. It is made of the cheap pressed sawdust type wood boards. The sink doesn't stay in place very well. No you cannot hook up water to it. Cannot believe people are asking this. The microwave makes a sound and has a light, the stove/oven makes a sound. It comes with some pots and pans. My 2 yr old enjoys it. | This is fairly easy to put together. The quality is good for the price, but the weak points are the door hinges. Kids get carried away and if pressed the wrong way, the hinge part of the door will break right off. Amazon support is great, however, and shipped us another unit for us to pick replacement parts and ship back the broken ones.",
    "Review Count": 1383,
    "Rating": "4.3",
    "Price": "$130.93"
  },
  {
    "sp_name": "Yalujumb Pretend Play Kitchen Appliances Toy Set with Coffee Maker Machine,Blender, Mixer and Toaster with Realistic Light and Sounds for Kids Ages 4-8",
    "Images": [
      {
        "url": "https://m.media-amazon.com/images/I/719X01IG2AL._AC_SL1500_.jpg"
      }
    ],
    "Image Link": "https://m.media-amazon.com/images/I/719X01IG2AL._AC_SL1500_.jpg",
    "sp_other_details": "Product information Product Dimensions 10 x 6.8 x 7 inches Item Weight 1.98 pounds Country of Origin China ASIN B09PBK2255 Manufacturer recommended age 36 months - 12 years Best Sellers Rank #3,724 in Toys & Games (See Top 100 in Toys & Games) #22 in Toy Kitchen Sets #39 in Preschool Kitchen Sets & Play Food Customer Reviews 4.3 4.3 out of 5 stars 1,114 ratings 4.3 out of 5 stars Department unisex-child Manufacturer YLJ Warranty & Support Product Warranty: For warranty information about this product, please click here Feedback Would you like to tell us about a lower price? Yalujumb Pretend Play Kitchen Appliances Toy Set with Coffee Maker Machine,Blender, Mixer and Toaster with Realistic Light and Sounds for Kids Ages 4-8 Share: Found a lower price? Let us know. Although we can't match every price reported, we'll use your feedback to ensure that our prices remain competitive. Where did you see a lower price? Website (Online) URL: Price: ($) Shipping cost: ($) Date of the price: 01 02 03 04 05 06 07 08 09 10 11 12 / 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 / Store (Offline) Store name: City: State: Please select province Please select province Price: ($) Date of the price: 01 02 03 04 05 06 07 08 09 10 11 12 / 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 / Please sign in to provide feedback. Submit Feedback",
    "Description": "Go to your orders and start the returnSelect the return methodShip it!",
    "sp_description": "Go to your orders and start the returnSelect the return methodShip it!",
    "sp_about": "1.Play Kitchen Appliances Set:Kids Kitchen Toys including coffee maker machine,mixer,toaster and blender with realistic light,cups and spoons. Assorted kitchen appliance toy set for your little one to have countless hours of fun and play.    2.Real Function: Mixer with Rotating Whip. Toaster with Timer and Pop-up Toast. Fillable Blender with Rotating Function.Coffee Maker for pretend play with sound effects.Kids can do same thing like in real cooking!    3.High-Quality & Child-friendly:Made of high-quality ABS Material.Bright Color and Durable,Smooth Edge without Burrs,Odor-free. It's a fun gift for Christmas, birthday or other special occasions. Surprise your smaller children with this Kitchen Appliances Set.    4.Endless Fun & Educational Kitchen Toy: This Comprehensive Pretend Play Kitchen Appliance Play Food Toy Set Provides Hours of Entertainment to Children Who Love to Imitate Adults Cooking. The Pretend Play Progress Trains Kids Eye-Hand Coornidation and Creative Imaginations.    5.Customer Satisfaction: Providing a 100% satisfaction experience is our main priority to our customers. Feel free to message us through “contact sellers” if products don't meet your expectations.",
    "Buy Link": "https://www.amazon.com/Yalujumb-Kitchen-Appliances-Pretend-Realistic/dp/B09PBK2255/ref=sr_1_9?keywords=Best+Play+Kitchen+for+Kids&qid=1701765527&sr=8-9",
    "Reviews": "The mechanics all work as discribed. I do think the coffee pot should be bigger . It is same size as the little coffee cups. I really won’t be able to rate appropiately tell my granddaughter gets ahold of it . It is a christmas gift | Bought these for my granddaughter and she loved them. Would have put 5 stars if some of them didn’t break. After about of month of use the motor burned out in the coffee maker. Very disappointed that the product broke already. | The toys are much smaller than I expected but my daughter loved them. I didn't give it 5 stars because there were no directions inside of the package. The operation of some of the appliances makes me assume you can put liquid in it but I didn't want to take that chance since it wasn't clearly stated. | I bought these to use for my preschool classroom. Within 2 days the toaster stopped making noise despite having fresh batteries and being the least frequently used. Honestly though the toaster was the loudest of the set and made a noise I’ve never heard come out of a real toaster in my life, like rocks through a blender, so I wasn’t too mad about it. The inner part of the blender that spins comes out every few minutes and the bottom of the blender bottle broke after about a week. Within two weeks one of the beaters on the mixer just falls out and the top of the coffee maker broke completely off the base. These are really cute toys, but better suited to calm children in a single or two child house. Not suited to heavy use in a child care setting. | These are adorable *little* toys. The coffee maker does not drip water into the pot fast at all. Thats the only downside. The mixer and blender are cute but hardly fit anything. I was looking for bigger ones like they sell at Walmart but overall they get the job done.",
    "Review Count": 1114,
    "Rating": "4.3",
    "Price": "$35.95"
  },
  {
    "sp_name": "Lil' Jumbl Kids Kitchen Set, Pretend Wooden Play Kitchen, Battery Operated Icemaker & Microwave with Realistic Sound, Pots & Pan Included - Charcoal",
    "Images": [
      {
        "url": "https://m.media-amazon.com/images/I/81VHGicxBWL._AC_SL1500_.jpg"
      }
    ],
    "Image Link": "https://m.media-amazon.com/images/I/81VHGicxBWL._AC_SL1500_.jpg",
    "sp_other_details": "Product information Product Dimensions 40.1 x 12.5 x 40.5 inches Item Weight 54.6 pounds ASIN B08T1NBZKF Item model number JUMWODKTCH Manufacturer recommended age 36 months - 12 years Best Sellers Rank #17,938 in Toys & Games (See Top 100 in Toys & Games) #115 in Toy Kitchen Sets #193 in Preschool Kitchen Sets & Play Food Customer Reviews 4.5 4.5 out of 5 stars 1,068 ratings 4.5 out of 5 stars Manufacturer Lil' Jumbl Warranty & Support Product Warranty: For warranty information about this product, please click here Feedback Would you like to tell us about a lower price? Lil' Jumbl Kids Kitchen Set, Pretend Wooden Play Kitchen, Battery Operated Icemaker & Microwave with Realistic Sound, Pots & Pan Included - Charcoal Share: Found a lower price? Let us know. Although we can't match every price reported, we'll use your feedback to ensure that our prices remain competitive. Where did you see a lower price? Website (Online) URL: Price: ($) Shipping cost: ($) Date of the price: 01 02 03 04 05 06 07 08 09 10 11 12 / 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 / Store (Offline) Store name: City: State: Please select province Please select province Price: ($) Date of the price: 01 02 03 04 05 06 07 08 09 10 11 12 / 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 / Please sign in to provide feedback. Submit Feedback",
    "Description": "Ships fromAmazonShips fromAmazonSold byDBROTHSold byDBROTHReturnsReturnable until Jan 31, 2024Returnable until Jan 31, 2024For the 2023 holiday season, eligible items purchased between November 1 and December 31, 2023 can be returned until January 31, 2024Read full return policyReturnsReturnable until Jan 31, 2024For the 2023 holiday season, eligible items purchased between November 1 and December 31, 2023 can be returned until January 31, 2024Read full return policyPaymentSecure transactionYour transaction is secureWe work hard to protect your security and privacy. Our payment security system encrypts your information during transmission. We don’t share your credit card details with third-party sellers, and we don’t sell your information to others.Learn morePaymentSecure transactionWe work hard to protect your security and privacy. Our payment security system encrypts your information during transmission. We don’t share your credit card details with third-party sellers, and we don’t sell your information to others.Learn moreSupportProduct support includedWhat's Product Support?In the event your product doesn't work as expected or you need help using it, Amazon offers free product support options such as live phone/chat with an Amazon associate, manufacturer contact information, step-by-step troubleshooting guides, and help videos. \n\nBy solving product issues, we help the planet by extending the life of products. Availability of support options differ by product and country.Learn moreSupportProduct support includedIn the event your product doesn't work as expected or you need help using it, Amazon offers free product support options such as live phone/chat with an Amazon associate, manufacturer contact information, step-by-step troubleshooting guides, and help videos. \n\nBy solving product issues, we help the planet by extending the life of products. Availability of support options differ by product and country.Learn moreDetails",
    "sp_description": "Ships fromAmazonShips fromAmazonSold byDBROTHSold byDBROTHReturnsReturnable until Jan 31, 2024Returnable until Jan 31, 2024For the 2023 holiday season, eligible items purchased between November 1 and December 31, 2023 can be returned until January 31, 2024Read full return policyReturnsReturnable until Jan 31, 2024For the 2023 holiday season, eligible items purchased between November 1 and December 31, 2023 can be returned until January 31, 2024Read full return policyPaymentSecure transactionYour transaction is secureWe work hard to protect your security and privacy. Our payment security system encrypts your information during transmission. We don’t share your credit card details with third-party sellers, and we don’t sell your information to others.Learn morePaymentSecure transactionWe work hard to protect your security and privacy. Our payment security system encrypts your information during transmission. We don’t share your credit card details with third-party sellers, and we don’t sell your information to others.Learn moreSupportProduct support includedWhat's Product Support?In the event your product doesn't work as expected or you need help using it, Amazon offers free product support options such as live phone/chat with an Amazon associate, manufacturer contact information, step-by-step troubleshooting guides, and help videos. \n\nBy solving product issues, we help the planet by extending the life of products. Availability of support options differ by product and country.Learn moreSupportProduct support includedIn the event your product doesn't work as expected or you need help using it, Amazon offers free product support options such as live phone/chat with an Amazon associate, manufacturer contact information, step-by-step troubleshooting guides, and help videos. \n\nBy solving product issues, we help the planet by extending the life of products. Availability of support options differ by product and country.Learn moreDetails",
    "sp_about": "About this item    SUPER REALISTIC SIGHTS & SOUNDS | Much More Than a Toy, Our Interactive Design Enables Meaningful STEM Learning as Boys & Girls Immerse Themselves in a Real Kitchen Setup | Elements Include Battery-Operated Ice Dispenser, Microwave with Buttons & Sounds, Clicking Knobs & More    BONUS TOOLS & ACCESSORIES | Tots Can Start Playing Straight Out the Box with [2] Free Stovetop Toys [Pot & Frying Pan] & Integrated Chalkboard for Jotting Down Ingredients & Other Creative Play | Kitchen Also Features a Real Towel Rack & Other Authentic Details Like Faucet, Sink Knobs, Door Handles & Cabinet Windows    THE ULTIMATE TRUE-TO-LIFE KITCHEN | All-in-One Kids Play Set Provides Hours of Fun, Educational Entertainment | See Little Ones Imagine, Explore, Learn & Pretend with a Variety of Lifelike Cooking & Cleaning Essentials Including a Refrigerator, Freezer, Ice Maker, Microwave, Oven, Stovetop, Sink, Dishwasher, Shelving, Towel Rack, Etc.    DURABLE WOOD CONSTRUCTION | High-Quality Furniture & Fixtures are Crafted of Heavy-Duty Composite Wood & Molded Plastic Parts for Utmost Safety & Ability to Withstand Years of Wear & Tear | Despite its Super Extensive Design, Playset is Fast & Easy to Assemble with All Hardware & Instructions Included [Requires Screwdriver]    FREESTANDING FOR ANY SETTING | Compact, Lightweight Play Kitchen is Effortless to Move & Maneuver, Weighing Under 60 Lbs. & Measuring 40.3” x 11.4” x 40.5” | Freestanding Style is Perfect for Play Room, Bedroom, Daycare & School Classroom with Built-In Shelving & Cabinetry for Accessory Storage",
    "Buy Link": "https://www.amazon.com/Jumbl-Realistic-Interactive-Chalkboard-Accessories/dp/B08T1NBZKF/ref=sr_1_31?keywords=Best+Play+Kitchen+for+Kids&qid=1701765540&sr=8-31",
    "Reviews": "We ordered this for my granddaughter’s birthday.  It would have been wonderful but for the fact that there is grease or some other kind of stains on multiple panels.  It also took us 2 1/2 hours to put together.  I thought for sure I’d be able to get the stains out but nope, nothing worked! When you’re spending so much on something, it shouldn’t be damaged! | There are several steps to assemble and many parts. Can be done with 2 people. Super cute when it’s all put together, but disappointing that one of the cabinet doors cracked so easily! My son is only 2.5yo and wasn’t being rough with the doors | In the video it showed a lot more features. This is basically just a storage box. | Our son really enjoyed this for the first three days. However one of the doors snapped off easily during typical play because the hinges allow for hyper extension. Also, the microwave battery compartment arrived sealed shut and could not be opened. I will be replacing this with a kitchen that is constructed for toddlers and their occasional curiosity and rambunctiousness. I think this product would work fine for a child who very carefully opens and closes things, orange sprinkles and someone who is looking for a very temporary toy that can be scrapped after a few weeks.  Another example of the craftsmanship is the door latches, they snap off very easily and prevent the doors from staying closed. If this sounds like your jam, go for it! | My son just turned four, but he’s pretty tall for his age, he like it, but I would have definitely picked another kitchen set that at least made some kind of noise. For the price the stove should light up or do something smh. The ice machine doesn’t work. The microwave doesn’t work. Imma add some extra things to it to make it a little bit more interesting. To assemble this thing……. Let me tell you, it doesn’t come preassembled. It wasn’t to bad but it did take about 3 hours, and it takes two people. It’s very sturdy because it’s all wood. If I had another chance to choose, it would definitely NOT be this one. Disappointed.",
    "Review Count": 1068,
    "Rating": "4.5",
    "Price": "$159.99"
  },
  {
    "sp_name": "ROBUD Wooden Play Kitchen Set for Kids Toddlers, Toy Kitchen Gift for Boys Girls, Age 3+",
    "Images": [
      {
        "url": "https://m.media-amazon.com/images/I/61+TVrElkSL._AC_SL1500_.jpg"
      }
    ],
    "Image Link": "https://m.media-amazon.com/images/I/61+TVrElkSL._AC_SL1500_.jpg",
    "sp_other_details": "Product information Product Dimensions 14 x 31.3 x 38.6 inches Item Weight 38.4 pounds Country of Origin China ASIN B0931ZVQ49 Item model number WCF14 Manufacturer recommended age 3 years and up Best Sellers Rank #13,338 in Toys & Games (See Top 100 in Toys & Games) #79 in Toy Kitchen Sets #144 in Preschool Kitchen Sets & Play Food Customer Reviews 4.5 4.5 out of 5 stars 324 ratings 4.5 out of 5 stars Release date April 21, 2021 Manufacturer ROBUD Warranty & Support Manufacturer’s warranty can be requested from customer service. Click here to make a request to customer service. Feedback Would you like to tell us about a lower price? ROBUD Wooden Play Kitchen Set for Kids Toddlers, Toy Kitchen Gift for Boys Girls, Age 3+ Share: Found a lower price? Let us know. Although we can't match every price reported, we'll use your feedback to ensure that our prices remain competitive. Where did you see a lower price? Website (Online) URL: Price: ($) Shipping cost: ($) Date of the price: 01 02 03 04 05 06 07 08 09 10 11 12 / 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 / Store (Offline) Store name: City: State: Please select province Please select province Price: ($) Date of the price: 01 02 03 04 05 06 07 08 09 10 11 12 / 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 / Please sign in to provide feedback. Submit Feedback",
    "Description": "Go to your orders and start the returnSelect the return methodShip it!",
    "sp_description": "Go to your orders and start the returnSelect the return methodShip it!",
    "sp_about": "About this item    THE PERFECT PLAY KITCHEN FOR YOUR LITTLE CHEF: Robud Kitchen Playset enhances creativity and fine motor skills. Made of solid wood, sturdy and durable construction.    CLASSIC GAME & TOY: This bright and colourful Wooden Kitchen Set and inspire early imaginative play by helping children to understand the concept of \"making\" food to eat.    REALISTIC FEATURES: The kitchen playset contains everything little gourmet chefs could need for preparing food. Turn the knobs to make clicking sound adds realism for pretend play.    IMAGINARY FUN: Role-playing game, allow your children give full play to their imagination and provide lots of fun. Develop children's interests and hobbies, exercise hands-on ability, and improve language expression and social skills in game.    GREAT GIFT: Kitchen toy promote dexterity, hand-eye coordination, This kitchen set is ideal for creative role-playing sessions and perfect gifts for kids.",
    "Buy Link": "https://www.amazon.com/ROBUD-Wooden-Kitchen-Toddlers-Girls/dp/B0931ZVQ49/ref=sr_1_3?keywords=Best+Play+Kitchen+for+Kids&qid=1701765527&sr=8-3",
    "Reviews": "I saw the reviews saying it took a lengthy time to put together, but it takes EVEN LONGER when not ONE of your pieces are labeled! I could not follow instructions since I had no clue what piece was what number since mine weren’t numbered. Thank goodness there was an assembly video on here or I would have had to return it. Also I was short 2 screws for this, which was fine since I had some on hand to use. | Missing 12. Sent 2 of the 14s | This was extremely difficult to put together and most of the screws were stripped or just did not fit. I ended up cutting some corners to put it together so that my son wasn’t disappointed. This is just an okay for now product and I would not repurchase or recommend. | I chose this soley because the description said it was solid wood. Which is super misleading. If I had known only 4-5 pieces were actually solid wood and the rest was particle wood, I would not have chosen this over the other options that were just as good but 50 dollars less. Also, as I was screwing in some pieces i could feel it not tighten and sort of just spin in place, and i just know this will not be holding up. Alot, of pieces were still wobbly upon completion. So dissappointing, but so much work to take it apart. Might just have my husband take it apart and return. Sigh. However, It is cute, but rather wide so will take up space. | Beautiful but came with damages.  I bought it one month before Christmas so I didn’t open it, when we were putting it together we found out it came with damages and it was too late to make a return because it was a few days before Christmas (and also it was the main gift so I needed to keep it)",
    "Review Count": 324,
    "Rating": "4.5",
    "Price": "$153.45"
  },
  {
    "sp_name": "BRINJOY Corner Play Kitchen for Kids, Wooden Toddler Kitchen Playset w/Faucet, Sink, Microwave, Oven, Apron, Blackboard, Storage Cabinets, Pretend Cooking Toys w/Sound & Light Gift for Ages 3+",
    "Images": [
      {
        "url": "https://m.media-amazon.com/images/I/71BVg4hY5JL._AC_SL1500_.jpg"
      }
    ],
    "Image Link": "https://m.media-amazon.com/images/I/71BVg4hY5JL._AC_SL1500_.jpg",
    "sp_other_details": "Product information Product Dimensions 39.5 x 35 x 38 inches Country of Origin China ASIN B09WTD9GFK Manufacturer recommended age 36 months - 10 years Best Sellers Rank #29,565 in Toys & Games (See Top 100 in Toys & Games) #207 in Toy Kitchen Sets #338 in Preschool Kitchen Sets & Play Food Customer Reviews 4.4 4.4 out of 5 stars 103 ratings 4.4 out of 5 stars Manufacturer BRINJOY Warranty & Support Product Warranty: For warranty information about this product, please click here Feedback Would you like to tell us about a lower price? BRINJOY Corner Play Kitchen for Kids, Wooden Toddler Kitchen Playset w/Faucet, Sink, Microwave, Oven, Apron, Blackboard, Storage Cabinets, Pretend Cooking Toys w/Sound & Light Gift for Ages 3+ Share: Found a lower price? Let us know. Although we can't match every price reported, we'll use your feedback to ensure that our prices remain competitive. Where did you see a lower price? Website (Online) URL: Price: ($) Shipping cost: ($) Date of the price: 01 02 03 04 05 06 07 08 09 10 11 12 / 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 / Store (Offline) Store name: City: State: Please select province Please select province Price: ($) Date of the price: 01 02 03 04 05 06 07 08 09 10 11 12 / 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 / Please sign in to provide feedback. Submit Feedback",
    "Description": "Ships fromBRINJOYShips fromBRINJOYSold byBRINJOYSold byBRINJOYReturnsReturnable until Jan 31, 2024Returnable until Jan 31, 2024For the 2023 holiday season, eligible items purchased between November 1 and December 31, 2023 can be returned until January 31, 2024Read full return policyReturnsReturnable until Jan 31, 2024For the 2023 holiday season, eligible items purchased between November 1 and December 31, 2023 can be returned until January 31, 2024Read full return policyPaymentSecure transactionYour transaction is secureWe work hard to protect your security and privacy. Our payment security system encrypts your information during transmission. We don’t share your credit card details with third-party sellers, and we don’t sell your information to others.Learn morePaymentSecure transactionWe work hard to protect your security and privacy. Our payment security system encrypts your information during transmission. We don’t share your credit card details with third-party sellers, and we don’t sell your information to others.Learn moreDetails",
    "sp_description": "Ships fromBRINJOYShips fromBRINJOYSold byBRINJOYSold byBRINJOYReturnsReturnable until Jan 31, 2024Returnable until Jan 31, 2024For the 2023 holiday season, eligible items purchased between November 1 and December 31, 2023 can be returned until January 31, 2024Read full return policyReturnsReturnable until Jan 31, 2024For the 2023 holiday season, eligible items purchased between November 1 and December 31, 2023 can be returned until January 31, 2024Read full return policyPaymentSecure transactionYour transaction is secureWe work hard to protect your security and privacy. Our payment security system encrypts your information during transmission. We don’t share your credit card details with third-party sellers, and we don’t sell your information to others.Learn morePaymentSecure transactionWe work hard to protect your security and privacy. Our payment security system encrypts your information during transmission. We don’t share your credit card details with third-party sellers, and we don’t sell your information to others.Learn moreDetails",
    "sp_about": "About this item    🍫【Realistic Cooking Experience】This pretend kitchen with sound and light will provide pleasure for kids. Featuring with full range of kitchen supplies, this play kitchen can meet all daily needs. The stove will automatically make the sound of frying and boiling when putting the pot on it. And the oven is equipped with light to create a more realistic role play experience for kids.    🍞【Extreme Large Storage Space】Our corner kitchen playset comes with 5 multi-storey cabinets with doors and a corner shelf at back for large storage space. In addition, 2 storage baskets can fit well into the compartments or use alone. 3 tiers shelves above the counter provide kids with enough space for placing spice bottles. Children can learn to organize their own toys after playing.    🍩【Durable & Kid-Friendly Design】 Constructed by solid MDF material, this corner kitchen playset for kids is stable and sturdy for long service life. The entire corner kitchen toy set is painted with kid-friendly paint so that your little kids can play without worry. All corners are rounded to provide all-round protection for children.    🍧【Realistic Role-play Experience】This oversized toddler play kitchen gives kids more space to play with their friends. The unique corner design can not only bring children fresh and realistic role play experience, but also save space in your home. The chalkboard and apron can make baby a real little cook -- They can put on apron and write the menu of the day on the chalkboard.    🎁【Great Gift for Kids Ages 3+ 】This kids kitchen toy set will be an ideal gift for girls and boys over 3 years old on Christmas, thanksgiving day and birthday. Children will develop and enhance creativity, hand-eye coordination and logical thinking in the process of playing. Our wooden kitchen toy set can be a good playmate for your kids as they grow up.",
    "Buy Link": "https://www.amazon.com/BRINJOY-Kitchen-Microwave-Blackboard-Cabinets/dp/B09WTD9GFK/ref=sr_1_20?keywords=Best+Play+Kitchen+for+Kids&qid=1701765540&sr=8-20",
    "Reviews": "Video Player is loading. Play Video Play Mute Current Time 0:00 / Duration 0:12 Loaded : 50.05% 0:00 Stream Type LIVE Seek to live, currently behind live LIVE Remaining Time - 0:12 1x Playback Rate Chapters Chapters Descriptions descriptions off , selected Captions Captions off , selected English Audio Track default , selected Fullscreen This is a modal window. Very cute idea of a kitchen I love the colors and how it looks but not sturdy at all and I have every single screw tightened as much as possible. The two sides of the kitchen are mostly just held together by little wooden pieces that constantly slip out and have to be pushed together again almost every hour . My child can’t play with it without the shelf in the oven falling and the two sides of the kitchen coming undone altogether so I’m basically just attaching the whole thing back together constantly. Unless you want to be fixing it constantly where it should already be sturdy like other kitchens then I wouldn’t waste the $200 on this one . It’s cute but would much rather another cute one that is sturdy enough for a child to play with without complications. | I opened the box and all the wood is pretty beat up and ALL the hardware is missing, literally not a single screw came with it. None of the accessories were in the box either. Basically what did come was a bunch of scuffed wood planks | There was not a stitch of instructions included and no pdf online | Difficult to assemble with poor directions and hard to see pictures. Can’t tell which way hardware facing. Not a quick and easy assemble. Arrived in damaged box. Damaged pieces inside. Missing pieces. Missing some hardware. Had to go to Home Depot to get parts to finish. Not worth $200 for all the trouble. Skip this one. Regret purchase. After all I have been through don’t feel like hassle of attempting to box back up to return. | I love the set expect my oven “glass” part and microwave “glass” part came VERYY scratched! I didn’t even notice till I already put the whole kitchen together i messaged the company and never responded back and I was willing to send pics for proof.",
    "Review Count": 104,
    "Rating": "4.4",
    "Price": "$189.99"
  },
  {
    "sp_name": "2-in-1Travel Luggage Turn into Play Kitchen, Kids Kitchen Playset Toys with 25+Play Food for Toddelers, Girls and Boys, Pretend Play, 25\" H x 9.8\" W x 20.8\" L",
    "Images": [
      {
        "url": "https://m.media-amazon.com/images/I/61Q185+p2BL._AC_SL1200_.jpg"
      }
    ],
    "Image Link": "https://m.media-amazon.com/images/I/61Q185+p2BL._AC_SL1200_.jpg",
    "sp_other_details": "N/A",
    "Description": "Go to your orders and start the returnSelect the return methodShip it!",
    "sp_description": "Go to your orders and start the returnSelect the return methodShip it!",
    "sp_about": "About this item    2-IN-1 PLAY KITCHEN SET FOR GIRLS: The design of portable storage box can not only set up a toy table, but also become a suitcase, which is convenient for children to travel and play in unlimited venues, so that children can develop the habit of of collecting and finishing from an early age.    KIDS KITCHEN PLAYSET FOR ROLE PLAY : This toy kitchen set contains 25+ accessories that everything children need, which includes dishes, cooking utensils, play pan and pot, vegetable and so on. It keeps your kid occupied for hours!    QUALITY & CHILD-FRIENDLY: Made of ABS, non-toxic; smooth and round edges, no burr, no sharp edges on every single accessory.    NEVER GO WRONG AS A GIFT -xa0The Play kitchen food makes a great addition to any play area. It is perfect for birthday party, Christmas, holiday or Impulse gift.    PRODUCT DIMENSIONS (L*W*H): 20.86*9.64*24.8 inches. Recommended for kids age 3 up",
    "Buy Link": "https://www.amazon.com/1Travel-Luggage-Kitchen-Playset-Toddelers/dp/B09Y1J5QP9/ref=sr_1_41?keywords=Best+Play+Kitchen+for+Kids&qid=1701765527&sr=8-41",
    "Reviews": "My day1 feeling. The item looks good, but the height or size didn't impress me by spending $36. My daughter is 3+ years old, and she has to sit and play with this toy and not a good fit to play by standing.  I had returned the item. I felt like a $20 price item. If the priced would come down by at least $10, it would be a perfect gift item for many Birthdays kids of ages 2 to 3. | Too flimsy | It was too small and not that great of quality | Bought it for a gift. Didn't explore the inside but the outside of the toy is pretty. Smaller than expected. :( | Bought this for my 8 year old, who still loves play kitchens, to take on a trip.  Well, that won’t be happening.  Concept is fantastic but items do not store inside it, you have to basically take it apart and put it back together again and a few pieces are already broke.  😩 Sheis trying really hard to take care of this set, but the struggle for her is real.",
    "Review Count": 76,
    "Rating": "4.1",
    "Price": "$36.98"
  },
  {
    "sp_name": "Joyin Pretend Kitchen Toys, Play Kitchen Accessories Set for Kids, Coffee Maker, Mixer, Toaster with Realistic Lights& Sounds, Kitchen Appliances Toys, Birthday Gift for Kids Ages 2 3 4 5",
    "Images": [
      {
        "url": "https://m.media-amazon.com/images/I/81zM7XNEr+L._AC_SL1500_.jpg"
      }
    ],
    "Image Link": "https://m.media-amazon.com/images/I/81zM7XNEr+L._AC_SL1500_.jpg",
    "sp_other_details": "Product information Product Dimensions 11.22 x 9.25 x 3.94 inches Item Weight 1.98 pounds Country of Origin China ASIN B0C8B7L7HN Item model number 15704 Manufacturer recommended age 3 years and up Best Sellers Rank #3,410 in Toys & Games (See Top 100 in Toys & Games) #20 in Toy Kitchen Sets #33 in Preschool Kitchen Sets & Play Food Customer Reviews 4.4 4.4 out of 5 stars 60 ratings 4.4 out of 5 stars Manufacturer JOYIN Warranty & Support Product Warranty: For warranty information about this product, please click here Feedback Would you like to tell us about a lower price? Joyin Pretend Kitchen Toys, Play Kitchen Accessories Set for Kids, Coffee Maker, Mixer, Toaster with Realistic Lights& Sounds, Kitchen Appliances Toys, Birthday Gift for Kids Ages 2 3 4 5 Share: Found a lower price? Let us know. Although we can't match every price reported, we'll use your feedback to ensure that our prices remain competitive. Where did you see a lower price? Website (Online) URL: Price: ($) Shipping cost: ($) Date of the price: 01 02 03 04 05 06 07 08 09 10 11 12 / 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 / Store (Offline) Store name: City: State: Please select province Please select province Price: ($) Date of the price: 01 02 03 04 05 06 07 08 09 10 11 12 / 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 / Please sign in to provide feedback. Submit Feedback",
    "Description": "1.Play Kitchen Appliances Set 2.Real Function 3.High-Quality & Child-friendly 4.Endless Fun & Educational Kitchen Toy",
    "sp_description": "1.Play Kitchen Appliances Set 2.Real Function 3.High-Quality & Child-friendly 4.Endless Fun & Educational Kitchen Toy",
    "sp_about": "Complete Kitchen Toys Set: Our product includes a coffee maker machine, toaster, and blender that will keep your child engaged for hours! The set also includes 2 knives, 2 spoons, 2 forks, 4 foods, 2 cups, 2 teacups, 2 plates, 2 saucers, and 2 toast, providing everything your little chef needs to create a delightful make-believe feast.    Real Function, Real Fun: The coffee maker machine dispenses water, the toaster pops up toast, and the blender actually rotates. These interactive features add a touch of realism to their imaginative play, making it even more exciting and enjoyable.    High-Quality & Child-Friendly: Our kitchen accessories are crafted with high-quality materials that meet the highest safety standards. They are designed with smooth edges and are perfectly sized for little hands. You can have peace of mind knowing that your child is playing with safe and child-friendly toys.    Endless Culinary Adventures: Let your child's creativity soar as they explore a world of culinary possibilities! They can use the coffee maker to brew a delicious cup of joe or the blender to whip up colorful smoothies. With the toaster, they can make crispy toast for breakfast, and the included utensils and plates allow them to serve up their culinary creations. It's a recipe for endless fun and imaginative play!    Nurture Cognitive Skills：Discover our incredible range of kitchen toys that foster children's hands-on and critical thinking skills. Watch as your little ones delve into a world of endless excitement and amusement, while actively developing their cognitive abilities. Experience the ultimate blend of fun and education with our innovative collection of educational toys!",
    "Buy Link": "https://www.amazon.com/Pretend-Accessories-Realistic-Appliances-Birthday/dp/B0C8B7L7HN/ref=sr_1_3?keywords=Best+Play+Kitchen+for+Kids&qid=1701765540&sr=8-3",
    "Reviews": "Very nice love the sound’s effects my toddlers loves them. The quality isn’t that great so I know they’ll break easy but still fun. A little pricey. | Make sure you check the size. | My son was so happy when he received this as a gift for his birthday. He makes us “coffee” everyday with his coffee pot. All of it runs on batteries so the coffee pot drips the coffee down and it also lights up. Only flaw design with it is that you have to hold the button down for it to work and it takes awhile for the water to start coming down. The blender really lights up and really spins. My son puts water in the little bowl and watches it turn. The toaster makes a ticking noise and pops the bread up when it’s done. The plates, plastic forks, spoons and fake food add a cute touch to it as well. The size of everything is small but it’s perfect for little hands and they fit well in his kitchen set. The color is pretty. He has fun every time he uses it but he does get annoyed with the coffee pot and having to hold the button down for it to work. I’m looking to replace it with something else that he he won’t have to hold down. | My daughter already has a little play kitchen and I thought this would be a great little addition for her. The work really well and she enjoys them a lot. Cheap plastic but their lasting well so far.",
    "Review Count": 60,
    "Rating": "4.4",
    "Price": "$25.99"
  },
  {
    "sp_name": "Disney Princess Style Collection Fresh Prep Gourmet Kitchen, Interactive Pretend Play Kitchen for Girls & Kids with Realistic Steam, Complete Meal Kit & 35+ Accessories",
    "Images": [
      {
        "url": "https://m.media-amazon.com/images/I/71LfbFbKICL._AC_SL1500_.jpg"
      }
    ],
    "Image Link": "https://m.media-amazon.com/images/I/71LfbFbKICL._AC_SL1500_.jpg",
    "sp_other_details": "N/A",
    "Description": "Go to your orders and start the returnSelect the return methodShip it!",
    "sp_description": "Go to your orders and start the returnSelect the return methodShip it!",
    "sp_about": "About this item    UPDATED MODERN DESIGN: Take play cooking to a whole new level with the Disney Princess Style Collection Fresh Prep Gourmet Kitchen!    REALISTIC “STEAM”: Includes an ALL NEW “steam” feature for the most realistic stovetop cooking experience with light up & sounds stovetop burner that recognizes the Pot and Fry pan with boiling and sizzling sounds.    OPEN ENDED PLAY FOR HOURS: Complete with 5 interactive appliances (Sound effects recipe card reading Tablet, Lights & Sounds Microwave, Lights Sounds, & “Steam” Stove Top, Sound Effects Faucet and Dishwasher)    INTERACTIVE MAJESTIC MEALS KIT: Pretend to prepare a gourmet meal with the included Pasta Night Kit which includes kitchen tools, play food pieces along with color changing noodles in icy water, and a recipe card that interacts with the interactive tablet    GREAT GIFT IDEA: Fresh Prep Gourmet Kitchen makes a great gift idea for bithday, holiday, or gift for Disney Princess fans    Includes 35+ Pretend Play Accessories: Pasta Night Meal Box, Interactive Recipe Card, Pasta Box, 1 Serving of Spaghetti, 3 Florets of Broccoli, 2 Garlic Bread Slices, 1 Serving of Meatballs, Can of Tomato Sauce, Fry Pan, Pot, Lid, Oven Rack, Fridge Shelf    Herb Garden Box, 3 Rosemary Sprigs, 3 Parsley Sprigs, 3 Basil Leaves, Trash Can, Olive Oil Bottle, Brush, Sponge, Cutting Board, Knife, Serving Spoon, Spatula, Ramekin, Dish Rack, Colander, 2 Dinner Plates, 2 Spoons, 2 Forks, 2 Knives, Glass, 2 Ice Cubes    \n   Adult assembly required. Requires 3 AA batteries, not included Suggested for Ages 3+   Show more",
    "Buy Link": "https://www.amazon.com/Disney-Princess-Collection-Gourmet-Kitchen/dp/B07KW6JRQ9/ref=sr_1_35?keywords=Best+Play+Kitchen+for+Kids&qid=1701765555&sr=8-35",
    "Reviews": "I don’t know if I received a defective piece or what but the stove doesn’t attach to the sink . It’s a long wire piece with the sink short piece no way it goes in. It pic it shows a short one which would of connected but I didn’t have that piece by the big wire cord to connect to sink",
    "Review Count": 16,
    "Rating": "4.1",
    "Price": "$199.99"
  },
  {
    "sp_name": "Veitch Fairytales Kids Kitchen Playset, Kids Play Food Cooking Games Pots and Pans BBQ Grill Toy Kitchen Accessories Set, 3+ Year Old Girl Boy Gift Ideas",
    "Images": [
      {
        "url": "https://m.media-amazon.com/images/I/81KyF--abwL._AC_SL1500_.jpg"
      }
    ],
    "Image Link": "https://m.media-amazon.com/images/I/81KyF--abwL._AC_SL1500_.jpg",
    "sp_other_details": "Product information Product Dimensions 11 x 6 x 7 inches Item Weight 2.01 pounds Country of Origin China ASIN B0C5J3R38C Item model number WQ918 Manufacturer recommended age 36 months - 3 years Best Sellers Rank #61,533 in Toys & Games (See Top 100 in Toys & Games) #375 in Toy Kitchen Sets #584 in Preschool Kitchen Sets & Play Food Customer Reviews 3.9 3.9 out of 5 stars 3 ratings 3.9 out of 5 stars Manufacturer SHANTOU VEITCH FAIRYTALES TRADING CO. , LTD. Warranty & Support Product Warranty: For warranty information about this product, please click here Feedback Would you like to tell us about a lower price? Veitch Fairytales Kids Kitchen Playset, Kids Play Food Cooking Games Pots and Pans BBQ Grill Toy Kitchen Accessories Set, 3+ Year Old Girl Boy Gift Ideas Share: Found a lower price? Let us know. Although we can't match every price reported, we'll use your feedback to ensure that our prices remain competitive. Where did you see a lower price? Website (Online) URL: Price: ($) Shipping cost: ($) Date of the price: 01 02 03 04 05 06 07 08 09 10 11 12 / 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 / Store (Offline) Store name: City: State: Please select province Please select province Price: ($) Date of the price: 01 02 03 04 05 06 07 08 09 10 11 12 / 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 / Please sign in to provide feedback. Submit Feedback",
    "Description": "Go to your orders and start the returnSelect the return methodShip it!",
    "sp_description": "Go to your orders and start the returnSelect the return methodShip it!",
    "sp_about": "kitchen set for kids    kids kitchen playset    play food",
    "Buy Link": "https://www.amazon.com/Veitch-Fairytales-Kitchen-Playset-Accessories/dp/B0C5J3R38C/ref=sr_1_42?keywords=Best+Play+Kitchen+for+Kids&qid=1701765527&sr=8-42",
    "Reviews": "My kids still had a blast with this did not realize that most of the food (except the pizza and the cutting food ) is really small like the tomato and orange is about the size of a small bounce ball 3+ maybe but if your little like putting stuff in their mouth still watch very very carefully or just get a different set with bigger food the pot pans and everything else was good and as I said my kids still have fun playing with it.",
    "Review Count": 3,
    "Rating": "3.9",
    "Price": "$50.99"
  },
  {
    "sp_name": "Teamson Kids Little Chef Atlanta Large Modular Wooden Play Kitchen with Interactive, Realistic Features, and 17 Kitchen Accessories, for 3yrs and up, White/Gold",
    "Images": [
      {
        "url": "https://m.media-amazon.com/images/I/61xaN+SZUKL._AC_SL1500_.jpg"
      }
    ],
    "Image Link": "https://m.media-amazon.com/images/I/61xaN+SZUKL._AC_SL1500_.jpg",
    "sp_other_details": "Product information Product Dimensions 14.65 x 61.1 x 34.37 inches Item Weight 61.5 pounds Country of Origin China ASIN B0CDX4SJRT Item model number TD-13850B Manufacturer recommended age 36 months - 12 years Best Sellers Rank #40,247 in Toys & Games (See Top 100 in Toys & Games) #255 in Toy Kitchen Sets #408 in Preschool Kitchen Sets & Play Food Customer Reviews 5.0 5.0 out of 5 stars 1 rating 5.0 out of 5 stars Release date July 8, 2023 Manufacturer Teamson Warranty & Support Manufacturer’s warranty can be requested from customer service. Click here to make a request to customer service. Feedback Would you like to tell us about a lower price? Teamson Kids Little Chef Atlanta Large Modular Wooden Play Kitchen with Interactive, Realistic Features, and 17 Kitchen Accessories, for 3yrs and up, White/Gold Share: Found a lower price? Let us know. Although we can't match every price reported, we'll use your feedback to ensure that our prices remain competitive. Where did you see a lower price? Website (Online) URL: Price: ($) Shipping cost: ($) Date of the price: 01 02 03 04 05 06 07 08 09 10 11 12 / 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 / Store (Offline) Store name: City: State: Please select province Please select province Price: ($) Date of the price: 01 02 03 04 05 06 07 08 09 10 11 12 / 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 / Please sign in to provide feedback. Submit Feedback",
    "Description": "The Little Chef Atlanta Play Kitchen is the perfect way for toddlers to explore their culinary creativity. With its modular design, stylish white and gold color scheme, and interactive features like clicking and turning oven knobs, a built-in icemaker, and a magnetic fridge door, this play kitchen is sure to be a hit with any toddler. It also comes with play food and pretend cookware for a complete play experience. All Teamson Kids products are certified safety tested, are lead and phthalates-free using non-toxic paint, is CPSIA compliant, and meet ASTM F963 standards. Teamson Kids gives flight to fun, discovery, and learning through beautifully crafted, safe, trend-right furniture pieces and toys for kids with a spirit of curiosity and adventure.",
    "sp_description": "The Little Chef Atlanta Play Kitchen is the perfect way for toddlers to explore their culinary creativity. With its modular design, stylish white and gold color scheme, and interactive features like clicking and turning oven knobs, a built-in icemaker, and a magnetic fridge door, this play kitchen is sure to be a hit with any toddler. It also comes with play food and pretend cookware for a complete play experience. All Teamson Kids products are certified safety tested, are lead and phthalates-free using non-toxic paint, is CPSIA compliant, and meet ASTM F963 standards. Teamson Kids gives flight to fun, discovery, and learning through beautifully crafted, safe, trend-right furniture pieces and toys for kids with a spirit of curiosity and adventure.",
    "sp_about": "About this item    THREE SEPARATE PIECES FOR THE PLAYROOM: This play kitchen includes a freestanding refrigerator, sink with counter space, and oven with counter space in a classic white finish with sleek gold accents    INTERACTIVE FEATURES MAKE IT REALISTIC: Built-in icemaker in the refrigerator, rotating faucet sprayer on the sink, and clicking coffee maker and oven knobs    PLENTY OF STORAGE SPACE: Refrigerator and oven open, storage pegs on the backsplash and side of the sink, and built-in storage by the stove and under the sink    (17) KITCHEN ACCESSORIES: a coffee pot, spatulas, pot & pan with lid, two seasoning bottles, a cup, three pretend ice cubes, 4 \"crackable\" eggs, two recycling bins and magnetic items for the refrigerator door    DIMENSIONS: 61.1 in. long, 14.65 in. wide, and 34.37 in. tall; made of durable engineered wood for years and years of playtime",
    "Buy Link": "https://www.amazon.com/Teamson-Kids-Interactive-Realistic-Accessories/dp/B0CDX4SJRT/ref=sr_1_2?keywords=Best+Play+Kitchen+for+Kids&qid=1701765527&sr=8-2",
    "Reviews": "",
    "Review Count": 1,
    "Rating": "5.0",
    "Price": "$299.99"
  }
]
save_to_airtable(all_product_details, "Best Play Kitchen for Kids", "173c12ee-54f0-4704-8ee9-9b43d4795ceb", "UK Baby")