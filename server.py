import os
from dotenv import load_dotenv

from typing import Type
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
import requests
import json
import logging
import re
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
import urllib.parse
from urllib.parse import quote
import openai


load_dotenv()
brwoserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")
airtable_key = os.getenv("AIRTABLE_API_KEY")

scrape_ant_key_US_baby_1 = os.getenv("SCRAPING_ANT_KEY_a1")
scrape_ant_key_US_baby_2 = os.getenv("SCRAPING_ANT_KEY_a2")
scrape_ant_key_UK_baby_1 = os.getenv("SCRAPING_ANT_KEY_b1")
scrape_ant_key_UK_baby_2 = os.getenv("SCRAPING_ANT_KEY_b2")
scrape_ant_key_US_beauty_1 = os.getenv("SCRAPING_ANT_KEY_c1")
scrape_ant_key_US_beauty_2 = os.getenv("SCRAPING_ANT_KEY_c2")

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
    
def load_cookies_for_scrapingant(path="cookies.json"):
    try:
        with open(path, "r") as file:
            cookies = json.load(file)
            # Format the cookies as 'name=value'
            formatted_cookies = ';'.join([f"{cookie['name']}={cookie['value']}" for cookie in cookies])
            # URL encode the formatted string
            encoded_cookies = urllib.parse.quote(formatted_cookies)
            return encoded_cookies
    except FileNotFoundError:
        return None  # No cookies file found
    
def save_cookie_string_to_file(cookie_string, path="cookies.json"):
    # Split the cookie string into individual cookies
    cookie_list = cookie_string.split(';')

    # Parse each cookie into a dictionary
    cookies_for_saving = []
    for cookie in cookie_list:
        name, value = cookie.strip().split('=', 1)  # Split on the first '='
        cookies_for_saving.append({"name": name, "value": value})

    # Save cookies to a file
    with open(path, "w") as file:
        json.dump(cookies_for_saving, file, indent=4)

    print(f"Cookies saved to {path}")


def search_amazon(query, type):
    product_details = []
    unique_amazon_results = []
    page_number = 1
    max_pages = 5  # Limit to prevent too many requests

    max_retries = 15
    retry_counter = 0
    wait_intervals = [10, 30]  # Wait times in seconds

    if os.path.exists("cookies.json"):
        os.remove("cookies.json")

    
    while len(unique_amazon_results) < 15 and page_number <= max_pages:
        print(f"Searching products for {query} - Page {page_number}")
        cookies = ""

        if os.path.exists("cookies.json"):
            cookies = load_cookies_for_scrapingant()
            cookies = "&cookies=" + cookies

        # Prepare the URL for ScrapingAnt API for each page
        encoded_query = quote(query)

        api_url = f"https://api.scrapingant.com/v2/extended?url=https%3A%2F%2F{get_amazon_url(type)}%2Fs%3Fk%3D{encoded_query}&page={page_number}&proxy_type=residential&x-api-key="+ get_scraping_agent_api_1(type) + "&proxy_country=" + get_amazon_proxy_country(type) + cookies
        print(api_url)
        
        try:
            response = requests.get(api_url)
            response.raise_for_status()

            # Parse response content to JSON
            data = response.json()

            # print("HTML: ", data.get('html', ''))
            print("Text: ", data.get('text', ''))
            print("Cookies: ", data.get('cookies', ''))


            if not os.path.exists("cookies.json"):
                save_cookie_string_to_file(data.get('cookies', ''))


            # Process the HTML content
            # Extract HTML content
            html_content = data.get('html', '')

            soup = BeautifulSoup(html_content, "html.parser")

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

                product_info['url'] = 'https://'+ get_amazon_url(type) + url_element['href'] if url_element else None

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
            retry_counter = 0

        except requests.RequestException as e:
            print(f"Error during requests to ScrapingAnt API: {e}")

            if retry_counter > max_retries:
                print("Max retries reached. Stopping the process.")
                break
            
            wait_time = wait_intervals[min(retry_counter - 1, len(wait_intervals) - 1)]
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    if len(unique_amazon_results) == 0 and page_number == 1:
        print("API Error")
        return None
    
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

async def scrape_website_ant(objective: str, url: str, type):

    max_retries = 6
    retry_counter = 0
    wait_intervals = [10, 30]  # Wait times in seconds


    try:
        print(f"Start Scraping {url}")

        cookies = ""
        if os.path.exists("cookies.json"):
            cookies = load_cookies_for_scrapingant()
            cookies = "&cookies=" + cookies


        time.sleep(random.uniform(3, 6))
        # Prepare the URL for ScrapingAnt API
        encoded_url = quote(url)
        api_url = f"https://api.scrapingant.com/v2/general?url={encoded_url}&x-api-key="+ get_scraping_agent_api_2(type) + "&proxy_country=" + get_amazon_proxy_country(type)  + cookies

        print(api_url)

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
            "Product Name": name,
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

        # print(product_details)

        if (name == "N/A"):
            print("Not a valid product content. Please find another product.")
        
            retry_counter += 1
            if retry_counter > max_retries:
                print("Max retries reached. Stopping the process.")
                

            wait_time = wait_intervals[min(retry_counter - 1, len(wait_intervals) - 1)]
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            
        else:
            retry_counter = 0
            return product_details


    except requests.RequestException as e:
        print(f"Error during requests to ScrapingAnt API: {e}")
        
    
        retry_counter += 1
        if retry_counter > max_retries:
            print("Max retries reached. Stopping the process.")
            return "Network error occurred."
    
        wait_time = wait_intervals[min(retry_counter - 1, len(wait_intervals) - 1)]
        print(f"Retrying in {wait_time} seconds...")
        time.sleep(wait_time)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return "An unexpected error occurred."
# asyncio.run(scrape_website_ant("","https://www.amazon.com/Adjustable-Memory-Pillow-Sleepers-Bamboo/dp/B0172GSQ7S"))

async def scrape_amazon_critical_reviews(asin, type):

    # Initial settings
    max_retries = 6
    retry_counter = 0
    wait_intervals = [10, 30]  # Wait times in seconds

    cookies = ""
    if os.path.exists("cookies.json"):
        cookies = load_cookies_for_scrapingant()
        cookies = "&cookies=" + cookies

    try:
        print(f"Start Scraping Critical Reviews for ASIN: {asin}")
        time.sleep(random.uniform(3, 6))

        # Construct the URL for the critical reviews page
        reviews_url = f"https://{get_amazon_url(type)}/product-reviews/{asin}/ref=cm_cr_arp_d_viewopt_sr?filterByStar=critical&pageNumber=1"
        encoded_url = quote(reviews_url)
        api_url = f"https://api.scrapingant.com/v2/general?url={encoded_url}&x-api-key="+ get_scraping_agent_api_1(type) + "&proxy_country=" + get_amazon_proxy_country(type) + cookies

        print(api_url)
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

        retry_counter = 0  # Reset retry counter after a successful request
        return reviews

    except requests.RequestException as e:
        print(f"Error during requests to ScrapingAnt API: {e}")

        retry_counter += 1
        if retry_counter > max_retries:
            print("Max retries reached. Stopping the process.")
            return "Network error occurred."

        wait_time = wait_intervals[min(retry_counter - 1, len(wait_intervals) - 1)]
        print(f"Retrying in {wait_time} seconds...")
        time.sleep(wait_time)
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return "An unexpected error occurred."

# asyncio.run(scrape_amazon_critical_reviews("B08DY55SQZ"))

def format_reviews_for_airtable(reviews):
    delimiter = " | "
    formatted_reviews = []

    for review in reviews:
        if isinstance(review, dict) and 'text' in review:
            formatted_reviews.append(review['text'])
        else:
            # Log the error or send a notification
            print(f"Invalid review format: {review}")

    formatted_reviews = delimiter.join(formatted_reviews)

    max_length = 10000
    delimiter = " | "
    formatted_reviews = []

    for review in reviews:
        if isinstance(review, dict) and 'text' in review:
            formatted_reviews.append(review['text'])
        else:
            # Log the error or send a notification
            print(f"Invalid review format: {review}")

    formatted_reviews = delimiter.join(formatted_reviews)

    max_length = 10000
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

    cookies = load_cookies_for_scrapingant()

    # content = agent({"input": "Top 10 " + query})
    search_results = search_amazon(query,type)

    print("Search Results", search_results)
    if search_results is None:
        print("Search failed. Exiting.")
        save_error_to_airtable(type)
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
            scrape_website_ant('Scrape product details', url,type))
        
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
                product_reviews = asyncio.run(scrape_amazon_critical_reviews(asin,type))
                formatted_reviews = format_reviews_for_airtable(product_reviews)
                
                product_details['Reviews'] = formatted_reviews
                product_details['Review Count'] = review_count
                product_details['Rating'] = rating

            if product_details.get('Images') and product_details['Images'][0].get('url').strip() not in ["", "N/A"]:
                product_details['Price'] = str(price)
                # product_details['Description'] = snippet

                all_product_details.append(product_details)
                product_count_with_images += 1

            product_details['Source'] = url
            product_details['Category'] = query
            product_details['batch_id'] = unique_id

            # Stop if we've gathered 10 products with image URLs
            if product_count_with_images >= 15:
                break
        else:
            print("Warning: product_details is not a dictionary")

    actual_content = all_product_details

    # print(actual_content)
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
    elif type == "US Beauty":
        return "appxmV9N7mo372EkX"
    else:
        raise ValueError("Invalid type specified")
    
    
def get_make_api_url(type):
    if type == "US Baby":
        return "https://hook.eu1.make.com/5uyqhpqm1beskwadyysebuvq23na7734"
    elif type == "UK Baby":
        return "https://hook.eu1.make.com/m198nfyf5pus5ijd4svjjyjbup9n2148"
    elif type == "US Beauty":
        return "https://hook.eu1.make.com/zrkuo3gwed1duqykaohastd573u1jat6"
    else:
        raise ValueError("Invalid type specified")
    
def get_scraping_agent_api_1(type):
    if type == "US Baby":
        return scrape_ant_key_US_baby_1
    elif type == "UK Baby":
        return scrape_ant_key_UK_baby_1
    elif type == "US Beauty":
        return scrape_ant_key_US_beauty_1
    else:
        raise ValueError("Invalid get_scraping_agent_api_1 type specified")
    
def get_scraping_agent_api_2(type):
    if type == "US Baby":
        return scrape_ant_key_US_baby_2
    elif type == "UK Baby":
        return scrape_ant_key_UK_baby_2
    elif type == "US Beauty":
        return scrape_ant_key_US_beauty_2
    else:
        raise ValueError("Invalid get_scraping_agent_api_1 type specified")
    
def get_amazon_url(type):
    if type == "UK Baby":
        return "www.amazon.co.uk"
    else:
        return "www.amazon.com"
    
def get_amazon_proxy_country(type):
    if type == "UK Baby":
        return "GB"
    else:
        return "US"
    
    

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

def save_error_to_airtable(type):
    API_URL = "https://api.airtable.com/v0/" + get_airtable_api_id(type) + "/Configuration"

    print("SAVING RECORDS TO AIRTABLE . . .")
    headers = {
        "Authorization": f"Bearer {airtable_key}",
        "Content-Type": "application/json"
    }

    data = {"fields": {"Status": "Error", "Remarks" : "No Products Found"}}
    requests.post(API_URL, headers=headers, json=data)

    print("Error message saved.")