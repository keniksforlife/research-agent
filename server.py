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
scrape_ant_key_YBC_CA = os.getenv("SCRAPING_ANT_KEY_c1")
scrape_ant_key_YBC_CA_2 = os.getenv("SCRAPING_ANT_KEY_c2")

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
        "model": "gpt-4o-mini",
        "messages": prompt_messages,
        "api_key": open_ai_key,
        # "headers": {"Openai-Version": "2020-11-07"},
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
    try:
        # Check if reviews is a list
        if not isinstance(reviews, list):
            # Log the error or handle the case appropriately
            print("Error: Reviews is None or not a list.")
            return ""  # Return an empty string or handle as needed

        delimiter = " | "
        formatted_reviews = []

        for review in reviews:
            if isinstance(review, dict) and 'text' in review:
                formatted_reviews.append(review['text'])
            else:
                # Optionally log the error or handle each invalid review format
                print(f"Warning: Invalid review format: {review}")

        # Join the reviews with the delimiter and ensure the length does not exceed max_length
        formatted_reviews = delimiter.join(formatted_reviews)
        max_length = 10000

        if len(formatted_reviews) > max_length:
            formatted_reviews = formatted_reviews[:max_length] + "..."

        return formatted_reviews

    except Exception as e:
        # Log the exception or handle it as needed
        print(f"An error occurred: {e}")
        # Optionally, return an error message or an empty string depending on how you want to handle errors
        return "An error occurred while formatting reviews."


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
    elif type == "US Beauty":
        return "appxmV9N7mo372EkX"
    elif type == "YBC CA":
        return "appkUBsGAJs1T91sk"
    else:
        raise ValueError("Invalid type specified")
    
    
def get_make_api_url(type):
    if type == "US Baby":
        return "https://hook.eu1.make.com/5uyqhpqm1beskwadyysebuvq23na7734"
    elif type == "UK Baby":
        return "https://hook.eu1.make.com/m198nfyf5pus5ijd4svjjyjbup9n2148"
    elif type == "US Beauty":
        return "https://hook.eu1.make.com/zrkuo3gwed1duqykaohastd573u1jat6"
    elif type == "YBC CA":
        return "https://hook.eu1.make.com/wbsq5f9697id895dpdlje9fls7xwjf6w"
    else:
        raise ValueError("Invalid type specified")
    
def get_scraping_agent_api_1(type):
    if type == "US Baby":
        return scrape_ant_key_US_baby_1
    elif type == "UK Baby":
        return scrape_ant_key_UK_baby_1
    elif type == "US Beauty":
        return scrape_ant_key_US_beauty_1
    elif type == "YBC CA":
        return scrape_ant_key_YBC_CA
    else:
        raise ValueError("Invalid get_scraping_agent_api_1 type specified")
    
def get_scraping_agent_api_2(type):
    if type == "US Baby":
        return scrape_ant_key_US_baby_2
    elif type == "UK Baby":
        return scrape_ant_key_UK_baby_2
    elif type == "US Beauty":
        return scrape_ant_key_US_beauty_2
    elif type == "YBC CA":
        return scrape_ant_key_YBC_CA_2
    else:
        raise ValueError("Invalid get_scraping_agent_api_1 type specified")
    
def get_amazon_url(type):
    if type == "UK Baby":
        return "www.amazon.co.uk"
    elif type == "YBC CA":
        return "www.amazon.ca"
    else:
        return "www.amazon.com"
    
def get_amazon_proxy_country(type):
    if type == "UK Baby":
        return "GB"
    if type == "YBC CA":
        return "CA"
    else:
        return "US"
    
a = [
    {
        "Product Name": "Leachco Snoogle Chic Jersey Total Body Pillow - Heather Gray",
        "Images": [
            {
                "url": "https://m.media-amazon.com/images/I/615cgdKouQL._AC_SL1000_.jpg"
            }
        ],
        "Image Link": "https://m.media-amazon.com/images/I/615cgdKouQL._AC_SL1000_.jpg",
        "sp_other_details": "Product information Technical Details Product Dimensions ‎149.23 x 64.77 x 19.69 cm; 2.27 kg Item model number ‎980Z-4-JGY Is discontinued by manufacturer ‎No Target gender ‎Unisex Material type ‎Cotton Blend Material free ‎BPA Free Care instructions ‎Hand Wash Only Additional product features ‎Washable Number of Items ‎1 Style ‎Body Pillow Batteries required ‎No Dishwasher safe ‎No Is portable ‎No Item Weight ‎2.27 kg Additional Information ASIN B003D7E2XI Customer Reviews 4.4 4.4 out of 5 stars 1,529 ratings 4.4 out of 5 stars Best Sellers Rank #21,703 in Home (See Top 100 in Home) #21 in Maternity Pillows Date First Available March 6 2012 Manufacturer Leachco Place of Business NEW YORK, NY, 10018 US Feedback Would you like to tell us about a lower price? Leachco Snoogle Chic Jersey Total Body Pillow - Heather Gray Share: Found a lower price? Let us know. Although we can't match every price reported, we'll use your feedback to ensure that our prices remain competitive. Where did you see a lower price? Fields with an asterisk * are required Price Availability Website (Online) URL *: Price incl. GST (CAD) *: Shipping cost (CAD): Date of the price (MM/DD/YYYY): 01 02 03 04 05 06 07 08 09 10 11 12 / 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 / Store (Offline) Store name *: Enter the store name where you found this product Enter the store name where you found this product City *: Province: Please select province Please select province Price incl. GST (CAD) *: Date of the price (MM/DD/YYYY): 01 02 03 04 05 06 07 08 09 10 11 12 / 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 / Submit Feedback Please sign in to provide feedback.",
        "Description": "Leachco Snoogle Chic Jersey Total Body Pillow Easy to use and give a comfortable feel, great product in a reasonable price",
        "sp_description": "Leachco Snoogle Chic Jersey Total Body Pillow Easy to use and give a comfortable feel, great product in a reasonable price",
        "sp_about": "About this item    The extra long midsection is the perfect width for total back or tummy support; Spot or Wipe Clean & Tumble Dry    Supports and aligns hips, back, neck, and tummy    One pillow that takes the place of multiple pillows    Comfort and support without the extra body heat    Patented design was developed by a Registered Nurse and Mom    \n›  See more product details",
        "Buy Link": "https://www.amazon.ca/Leachco-Snoogle-Jersey-Total-Pillow/dp/B003D7E2XI/ref=sr_1_32?dib=eyJ2IjoiMSJ9.q9kApQiMqmEEI1nTAzCebXrSHTbJHBwHS0tO_v-UHEwNUvgnzRfXeR7GNcXBf_NjssAYQgqK69I0bvlIB0ZK4xmbhl8hqxCfXsGhJzbILzjwOzuDAoXJO8UomXmSOoctcx7CDPcknlfDcif0ackJ_SY4rYBuPOte_ukV673Ysl5RqHCBWRjFdsNObxSjoedXUaTyI9GIGKs3U6vGu-7lm9hnJ0qhK_LwRIt7uZF3WARHiVyqMDZMi_2_36FFQCTtuYAFgClJbKaDflSY-xkVH40wFg-H4zFIAcbGgyit8_k.DFiun5ZfPuE5_6U89Hugoj01cAA__mbSc8i1WxnDj4g&dib_tag=se&keywords=Pregnancy+Pillows&qid=1733303700&sr=8-32",
        "Reviews": "",
        "Review Count": 1529,
        "Rating": "4.4",
        "Price": "$79.97",
        "Source": "https://www.amazon.ca/Leachco-Snoogle-Jersey-Total-Pillow/dp/B003D7E2XI/ref=sr_1_32?dib=eyJ2IjoiMSJ9.q9kApQiMqmEEI1nTAzCebXrSHTbJHBwHS0tO_v-UHEwNUvgnzRfXeR7GNcXBf_NjssAYQgqK69I0bvlIB0ZK4xmbhl8hqxCfXsGhJzbILzjwOzuDAoXJO8UomXmSOoctcx7CDPcknlfDcif0ackJ_SY4rYBuPOte_ukV673Ysl5RqHCBWRjFdsNObxSjoedXUaTyI9GIGKs3U6vGu-7lm9hnJ0qhK_LwRIt7uZF3WARHiVyqMDZMi_2_36FFQCTtuYAFgClJbKaDflSY-xkVH40wFg-H4zFIAcbGgyit8_k.DFiun5ZfPuE5_6U89Hugoj01cAA__mbSc8i1WxnDj4g&dib_tag=se&keywords=Pregnancy+Pillows&qid=1733303700&sr=8-32",
        "Category": "Pregnancy Pillows",
        "batch_id": "6500cc11-781b-4c6a-8794-bfe488464b1b"
    },
    {
        "Product Name": "BYRIVER 39inch Pink Purple Body Pillow for Women Girl Side Sleepers, C Shaped Pregnancy Pillow, Gifts for New Mom (ZIL)",
        "Images": [
            {
                "url": "https://m.media-amazon.com/images/I/612-l4Fo-2L._AC_SL1500_.jpg"
            }
        ],
        "Image Link": "https://m.media-amazon.com/images/I/612-l4Fo-2L._AC_SL1500_.jpg",
        "sp_other_details": "Product information Technical Details Brand ‎BYRIVER Colour ‎Pink Purple Product Dimensions ‎99 x 30 x 20 cm; 998 g Material ‎Cotton Special Features ‎Portable, Washable Item Weight ‎998 g Additional Information ASIN B0BFWR162R Customer Reviews 4.3 4.3 out of 5 stars 742 ratings 4.3 out of 5 stars Best Sellers Rank #16,797 in Home (See Top 100 in Home) #18 in Maternity Pillows Date First Available Sept. 20 2022 Feedback Would you like to tell us about a lower price? BYRIVER 39inch Pink Purple Body Pillow for Women Girl Side Sleepers, C Shaped Pregnancy Pillow, Gifts for New Mom (ZIL) Share: Found a lower price? Let us know. Although we can't match every price reported, we'll use your feedback to ensure that our prices remain competitive. Where did you see a lower price? Fields with an asterisk * are required Price Availability Website (Online) URL *: Price incl. GST (CAD) *: Shipping cost (CAD): Date of the price (MM/DD/YYYY): 01 02 03 04 05 06 07 08 09 10 11 12 / 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 / Store (Offline) Store name *: Enter the store name where you found this product Enter the store name where you found this product City *: Province: Please select province Please select province Price incl. GST (CAD) *: Date of the price (MM/DD/YYYY): 01 02 03 04 05 06 07 08 09 10 11 12 / 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 / Submit Feedback Please sign in to provide feedback.",
        "Description": "For most side sleepers and back sleepers, knee pain and lower back pain is so frustrating. Now you do not need any fancy devices to help prevent knee pain that happens when you sleep. Just hug BYRIVER Body Pillow and place it between your knees once you are ready to go to sleep. BYRIVER Full Body Pillow has a contoured shape that aligns with the shoulder, back, and leg preventing knee rubs against each other and relieving any type of nerve tension.",
        "sp_description": "For most side sleepers and back sleepers, knee pain and lower back pain is so frustrating. Now you do not need any fancy devices to help prevent knee pain that happens when you sleep. Just hug BYRIVER Body Pillow and place it between your knees once you are ready to go to sleep. BYRIVER Full Body Pillow has a contoured shape that aligns with the shoulder, back, and leg preventing knee rubs against each other and relieving any type of nerve tension.",
        "sp_about": "About this item    BODY SUPPORT: BYRIVER C J shaped full body pillow helps support your arm,belly,legs,and knees to ease discomfort associated with pregnancy,maternity,post-surgery,sciatica,fibromyalgia,and joint pain.Reduce lower back pressure and improve your body's spinal alignment.Make you sleep better.    NOT JUST FOR PREGNANT WOMEN-This full body pillow is perfect for anyone needing more support,recovering from surgery,or tired of having to use separate pillows to support their arms,legs,calf,and knees when sleeping.    100% COTTON PILLOWCASE: Washable breathable cotton pillow cover,soft,cozy and cool,ideal for hot sleepers.Filled with 2.2lbs superfine microfiber,fluffy,light weight,but supportive.    PORTABLE: this body pillow does not huge like other pregnancy pillows,and occupies too many bed places.It is super convenient to hug and turn around at night,a great helper for side sleepers.    VERSATILE: Buckle and it will become a nursing pillow,baby support pillow,learn-to-sit pillow,and positioner.The invisible zipper and cotton button knot we use make the entire pillow soft,smooth,and skin-friendly.    \n ›  See more product details",
        "Buy Link": "https://www.amazon.ca/sspa/click?ie=UTF8&spc=MTo2MDgwNzY4MTA4ODE3MTQ4OjE3MzMzMDM3MDA6c3BfbXRmOjIwMDEyMDgyMTE5NzI5ODo6MDo6&url=%2FBYRIVER-39inch-Purple-Sleepers-Pregnancy%2Fdp%2FB0BFWR162R%2Fref%3Dsr_1_23_sspa%3Fdib%3DeyJ2IjoiMSJ9.q9kApQiMqmEEI1nTAzCebXrSHTbJHBwHS0tO_v-UHEwNUvgnzRfXeR7GNcXBf_NjssAYQgqK69I0bvlIB0ZK4xmbhl8hqxCfXsGhJzbILzjwOzuDAoXJO8UomXmSOoctcx7CDPcknlfDcif0ackJ_SY4rYBuPOte_ukV673Ysl5RqHCBWRjFdsNObxSjoedXUaTyI9GIGKs3U6vGu-7lm9hnJ0qhK_LwRIt7uZF3WARHiVyqMDZMi_2_36FFQCTtuYAFgClJbKaDflSY-xkVH40wFg-H4zFIAcbGgyit8_k.DFiun5ZfPuE5_6U89Hugoj01cAA__mbSc8i1WxnDj4g%26dib_tag%3Dse%26keywords%3DPregnancy%2BPillows%26qid%3D1733303700%26sr%3D8-23-spons%26sp_csd%3Dd2lkZ2V0TmFtZT1zcF9tdGY%26psc%3D1",
        "Reviews": "",
        "Review Count": 742,
        "Rating": "4.3",
        "Price": "$54.99",
        "Source": "https://www.amazon.ca/sspa/click?ie=UTF8&spc=MTo2MDgwNzY4MTA4ODE3MTQ4OjE3MzMzMDM3MDA6c3BfbXRmOjIwMDEyMDgyMTE5NzI5ODo6MDo6&url=%2FBYRIVER-39inch-Purple-Sleepers-Pregnancy%2Fdp%2FB0BFWR162R%2Fref%3Dsr_1_23_sspa%3Fdib%3DeyJ2IjoiMSJ9.q9kApQiMqmEEI1nTAzCebXrSHTbJHBwHS0tO_v-UHEwNUvgnzRfXeR7GNcXBf_NjssAYQgqK69I0bvlIB0ZK4xmbhl8hqxCfXsGhJzbILzjwOzuDAoXJO8UomXmSOoctcx7CDPcknlfDcif0ackJ_SY4rYBuPOte_ukV673Ysl5RqHCBWRjFdsNObxSjoedXUaTyI9GIGKs3U6vGu-7lm9hnJ0qhK_LwRIt7uZF3WARHiVyqMDZMi_2_36FFQCTtuYAFgClJbKaDflSY-xkVH40wFg-H4zFIAcbGgyit8_k.DFiun5ZfPuE5_6U89Hugoj01cAA__mbSc8i1WxnDj4g%26dib_tag%3Dse%26keywords%3DPregnancy%2BPillows%26qid%3D1733303700%26sr%3D8-23-spons%26sp_csd%3Dd2lkZ2V0TmFtZT1zcF9tdGY%26psc%3D1",
        "Category": "Pregnancy Pillows",
        "batch_id": "6500cc11-781b-4c6a-8794-bfe488464b1b"
    },
    {
        "Product Name": "SLIGUY Pregnancy Pillow Cover G Shaped, 57-Inch Replacement Pillowcase, Used for Maternity Pillows, 100% Velvet, Double Zipper Stretch Fabric, Super Soft, Universal Type, (G Blue)",
        "Images": [
            {
                "url": "https://m.media-amazon.com/images/I/71DdIMRs34L._AC_SL1500_.jpg"
            }
        ],
        "Image Link": "https://m.media-amazon.com/images/I/71DdIMRs34L._AC_SL1500_.jpg",
        "sp_other_details": "Product information Technical Details Brand ‎SLIGUY Model Number ‎1 Colour ‎Blue-g Product Dimensions ‎127 x 152.4 x 0.1 cm; 306.17 g Material ‎Velvet Special Features ‎Washable Item Weight ‎306 g Additional Information ASIN B09FPTXT8M Customer Reviews 4.4 4.4 out of 5 stars 223 ratings 4.4 out of 5 stars Best Sellers Rank #40,431 in Home (See Top 100 in Home) #25 in Maternity Pillows Date First Available Oct. 5 2021 Manufacturer SLIGUY Place of Business SLIGUY Feedback Would you like to tell us about a lower price? SLIGUY Pregnancy Pillow Cover G Shaped, 57-Inch Replacement Pillowcase, Used for Maternity Pillows, 100% Velvet, Double Zipper Stretch Fabric, Super Soft, Universal Type, (G Blue) Share: Found a lower price? Let us know. Although we can't match every price reported, we'll use your feedback to ensure that our prices remain competitive. Where did you see a lower price? Fields with an asterisk * are required Price Availability Website (Online) URL *: Price incl. GST (CAD) *: Shipping cost (CAD): Date of the price (MM/DD/YYYY): 01 02 03 04 05 06 07 08 09 10 11 12 / 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 / Store (Offline) Store name *: Enter the store name where you found this product Enter the store name where you found this product City *: Province: Please select province Please select province Price incl. GST (CAD) *: Date of the price (MM/DD/YYYY): 01 02 03 04 05 06 07 08 09 10 11 12 / 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 / Submit Feedback Please sign in to provide feedback.",
        "Description": "Pillow Cover",
        "sp_description": "Pillow Cover",
        "sp_about": "About this item    【G-Shaped Maternity Pillow Case】G-shaped pregnancy pillow cover and body pillowcase specially designed for pregnant women’s sleep, soft and durable, top quality, suitable for 57 ''x 40'' x 27'' maternity pillows. The body pillowcase can provide a fit covering, create a comfortable fit, provide a comfortable feeling for pregnant women's sleep and maintain bulkiness.    【Skin-Friendly and Comfortable Material】 SLIGUY skin-friendly and comfortable pregnancy pillowcase/maternity body pillowcase is made of 100% velvet material that is soft and skin-friendly, and the fabric is comfortable and breathable, bringing the ultimate comfort and quality sleep experience to pregnant mothers. The order only includes pillowcases, not pillows.    【Double Zipper Design, Easy to Disassemble】The double-open smooth large zipper can easily fill in and take out your pregnant pillows, which is very important for washing or regular maintenance. It is equipped with a zipper hidden pocket, which is easier for daily use, and the double-needle seam is more firm and suitable for long-term use.    【Machine Wash Safe Material】100% velvet maternity pillowcase supports machine wash and dryer. This maternity pillowcase for sleeping has long-lasting strength, anti-shrinkage, and anti-wrinkle properties, and has a noticeably soft feel. Tests have shown that SLIGUY maternity pillowcases can withstand hundreds of washings.    【Guide for Use】It is recommended that you use a pillow and two pillowcases when using the pillow for pregnant women. This way your sleep or daily rest cycle will not be disturbed. If you have any other questions, you can contact us at any time.    \n ›  See more product details",
        "Buy Link": "https://www.amazon.ca/SLIGUY-Pregnancy-Replacement-Pillowcase-Maternity/dp/B09FPTXT8M/ref=sr_1_55?dib=eyJ2IjoiMSJ9.q9kApQiMqmEEI1nTAzCebXrSHTbJHBwHS0tO_v-UHEwNUvgnzRfXeR7GNcXBf_NjssAYQgqK69I0bvlIB0ZK4xmbhl8hqxCfXsGhJzbILzjwOzuDAoXJO8UomXmSOoctcx7CDPcknlfDcif0ackJ_SY4rYBuPOte_ukV673Ysl5RqHCBWRjFdsNObxSjoedXUaTyI9GIGKs3U6vGu-7lm9hnJ0qhK_LwRIt7uZF3WARHiVyqMDZMi_2_36FFQCTtuYAFgClJbKaDflSY-xkVH40wFg-H4zFIAcbGgyit8_k.DFiun5ZfPuE5_6U89Hugoj01cAA__mbSc8i1WxnDj4g&dib_tag=se&keywords=Pregnancy+Pillows&qid=1733303700&sr=8-55",
        "Reviews": "",
        "Review Count": 223,
        "Rating": "4.4",
        "Price": "$29.99",
        "Source": "https://www.amazon.ca/SLIGUY-Pregnancy-Replacement-Pillowcase-Maternity/dp/B09FPTXT8M/ref=sr_1_55?dib=eyJ2IjoiMSJ9.q9kApQiMqmEEI1nTAzCebXrSHTbJHBwHS0tO_v-UHEwNUvgnzRfXeR7GNcXBf_NjssAYQgqK69I0bvlIB0ZK4xmbhl8hqxCfXsGhJzbILzjwOzuDAoXJO8UomXmSOoctcx7CDPcknlfDcif0ackJ_SY4rYBuPOte_ukV673Ysl5RqHCBWRjFdsNObxSjoedXUaTyI9GIGKs3U6vGu-7lm9hnJ0qhK_LwRIt7uZF3WARHiVyqMDZMi_2_36FFQCTtuYAFgClJbKaDflSY-xkVH40wFg-H4zFIAcbGgyit8_k.DFiun5ZfPuE5_6U89Hugoj01cAA__mbSc8i1WxnDj4g&dib_tag=se&keywords=Pregnancy+Pillows&qid=1733303700&sr=8-55",
        "Category": "Pregnancy Pillows",
        "batch_id": "6500cc11-781b-4c6a-8794-bfe488464b1b"
    },
    {
        "Product Name": "Chilling Home Full Body Pillows for Adults with Washable Removable Pillowcase, Quilted Long Bed Pillows for Sleeping, Large Pillow for Pregnancy Women and Side Sleepers",
        "Images": [
            {
                "url": "https://m.media-amazon.com/images/I/71oChKjFYRL._AC_SL1500_.jpg"
            }
        ],
        "Image Link": "https://m.media-amazon.com/images/I/71oChKjFYRL._AC_SL1500_.jpg",
        "sp_other_details": "Product information Technical Details Brand ‎Chilling Home Model Number ‎CABPR54JWG01 Colour ‎White-golden Product Dimensions ‎137.16 x 53.34 x 0.1 cm; 2.58 kg Material ‎Polyester Special Features ‎Removable Cover Item Weight ‎2.58 kg Additional Information ASIN B0CWNHSRP1 Customer Reviews 4.5 4.5 out of 5 stars 60 ratings 4.5 out of 5 stars Best Sellers Rank #3,039 in Home (See Top 100 in Home) #4 in Body Pillows Date First Available Feb. 28 2024 Manufacturer Chilling Home Place of Business Contact Chilling Home Customer Service Team Via Order Information Feedback Would you like to tell us about a lower price? Chilling Home Full Body Pillows for Adults with Washable Removable Pillowcase, Quilted Long Bed Pillows for Sleeping, Large Pillow for Pregnancy Women and Side Sleepers Share: Found a lower price? Let us know. Although we can't match every price reported, we'll use your feedback to ensure that our prices remain competitive. Where did you see a lower price? Fields with an asterisk * are required Price Availability Website (Online) URL *: Price incl. GST (CAD) *: Shipping cost (CAD): Date of the price (MM/DD/YYYY): 01 02 03 04 05 06 07 08 09 10 11 12 / 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 / Store (Offline) Store name *: Enter the store na\ninfo\nme where you found this product Enter the store name where you found this product City *: Province: Please select province Please select province Price incl. GST (CAD) *: Date of the price (MM/DD/YYYY): 01 02 03 04 05 06 07 08 09 10 11 12 / 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 / Submit Feedback Please sign in to provide feedback.",
        "Description": "BrandChilling HomeSpecial featureRemovable CoverColourWhite-goldenSizeX-LargeFill materialPolyesterShapeRectangularAge range (description)AdultRecommended uses for productBack Sleeper, Multi Position Sleeper, Sleeping, Stomach SleeperUse forWhole BodyMaterialPolyester",
        "sp_description": "BrandChilling HomeSpecial featureRemovable CoverColourWhite-goldenSizeX-LargeFill materialPolyesterShapeRectangularAge range (description)AdultRecommended uses for productBack Sleeper, Multi Position Sleeper, Sleeping, Stomach SleeperUse forWhole BodyMaterialPolyester",
        "sp_about": "About this item    🛒【Large Size】54*21 Inch oreiller de corps, fit for most beds and body sizes, this body pillows for adults can offer you full body support, fit for pregnant women or side sleepers, and anyone who need a hug    ☕【Luxury Design】Luxurious and exquisite embroidery quilted Long Body Pillow, the unique design makes the pillow more fancy, decorative and elegant. Which can also be your home decor or gift for friends    💧【Easy Cleaning】An extra pillowcase with this body pillow for adults, the pillowcase is envelope closure, easy to put on and take off, machine washable    🎀【Comfortable and Supportive】New upgraded breathable polyester fabric with high rebound polyester filling, after dozens of tests, get the perfect filling amount, neither clumping, but also provides full support, your best cuddle pillow for both adults and child    🎉【How to Fluffy Up】We use the Eco-Friendly vacuum package, maybe this body pillow is not very fluffy and supportive. So please press, squeeze and fold it to get it back to its original full fluffy shape. Second, please allow 72 hours for the pillow to expand, then you can get your pillow in a better situation    \n ›  See more product details",
        "Buy Link": "https://www.amazon.ca/Chilling-Home-Removable-Pillowcase-Pregnancy/dp/B0CWNHSRP1/ref=sr_1_54?dib=eyJ2IjoiMSJ9.q9kApQiMqmEEI1nTAzCebXrSHTbJHBwHS0tO_v-UHEwNUvgnzRfXeR7GNcXBf_NjssAYQgqK69I0bvlIB0ZK4xmbhl8hqxCfXsGhJzbILzjwOzuDAoXJO8UomXmSOoctcx7CDPcknlfDcif0ackJ_SY4rYBuPOte_ukV673Ysl5RqHCBWRjFdsNObxSjoedXUaTyI9GIGKs3U6vGu-7lm9hnJ0qhK_LwRIt7uZF3WARHiVyqMDZMi_2_36FFQCTtuYAFgClJbKaDflSY-xkVH40wFg-H4zFIAcbGgyit8_k.DFiun5ZfPuE5_6U89Hugoj01cAA__mbSc8i1WxnDj4g&dib_tag=se&keywords=Pregnancy+Pillows&qid=1733303700&sr=8-54",
        "Reviews": "",
        "Review Count": 60,
        "Rating": "4.5",
        "Price": "$49.98",
        "Source": "https://www.amazon.ca/Chilling-Home-Removable-Pillowcase-Pregnancy/dp/B0CWNHSRP1/ref=sr_1_54?dib=eyJ2IjoiMSJ9.q9kApQiMqmEEI1nTAzCebXrSHTbJHBwHS0tO_v-UHEwNUvgnzRfXeR7GNcXBf_NjssAYQgqK69I0bvlIB0ZK4xmbhl8hqxCfXsGhJzbILzjwOzuDAoXJO8UomXmSOoctcx7CDPcknlfDcif0ackJ_SY4rYBuPOte_ukV673Ysl5RqHCBWRjFdsNObxSjoedXUaTyI9GIGKs3U6vGu-7lm9hnJ0qhK_LwRIt7uZF3WARHiVyqMDZMi_2_36FFQCTtuYAFgClJbKaDflSY-xkVH40wFg-H4zFIAcbGgyit8_k.DFiun5ZfPuE5_6U89Hugoj01cAA__mbSc8i1WxnDj4g&dib_tag=se&keywords=Pregnancy+Pillows&qid=1733303700&sr=8-54",
        "Category": "Pregnancy Pillows",
        "batch_id": "6500cc11-781b-4c6a-8794-bfe488464b1b"
    },
    {
        "Product Name": "Annamite Range Premium Soft Body Pillow, Hypoallergenic, Extra Fluffy, Durable, Includes Envelope-Style Silky Designer Bamboo Pillowcase (Grey) and Large Pillow Bag",
        "Images": [
            {
                "url": "https://m.media-amazon.com/images/I/61SIE9dCC5L._AC_SL1500_.jpg"
            }
        ],
        "Image Link": "https://m.media-amazon.com/images/I/61SIE9dCC5L._AC_SL1500_.jpg",
        "sp_other_details": "Product information Technical Details Brand ‎Annamite Range Model Number ‎ARBP01G Colour ‎Grey Product Dimensions ‎139 x 50 x 0.1 cm; 2.16 kg Material ‎Silk, Polyester Casing, Rayon from bamboo, 1800g Microfiber filling Special Features ‎Plushed with Generous Filling (1800g microfiber), Designer Envelope Style Closure, Breathable and Quality Materials, Hypoallergenic giving an irritation-free rest., Breathable and cool bamboo fabric that keeps you dry and cool throughout the night. Item Weight ‎2.16 kg Additional Information ASIN B0CWZFVY7D Customer Reviews 4.5 4.5 out of 5 stars 18 ratings 4.5 out of 5 stars Best Sellers Rank #255,761 in Home (See Top 100 in Home) #642 in Bed Pillow Pillowcases Date First Available March 3 2024 Manufacturer Annamite Range Place of Business Annamite Range Feedback Would you like to tell us about a lower price? Annamite Range Premium Soft Body Pillow, Hypoallergenic, Extra Fluffy, Durable, Includes Envelope-Style Silky Designer Bamboo Pillowcase (Grey) and Large Pillow Bag Share: Found a lower price? Let us know. Although we can't match every price reported, we'll use your feedback to ensure that our prices remain competitive. Where did you see a lower price? Fields with an asterisk * are required Price Availability Website (Online) URL *: Price incl. GST (CAD) *: Shipping cost (CAD): Date of the price (MM/DD/YYYY): 01 02 03 04 05 06 07 08 09 10 11 12 / 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 / Store (Offline) Store name *: Enter the store name where you found this product Enter the store name where you found this product City *: Province: Please select province Please select province Price incl. GST (CAD) *: Date of the price (MM/DD/YYYY): 01 02 03 04 05 06 07 08 09 10 11 12 / 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 / Submit Feedback Please sign in to provide feedback.",
        "Description": "BrandAnnamite RangeSpecial featurePlushed with Generous Filling (1800g microfiber), Designer Envelope Style Closure, Breathable and Quality Materials, Hypoallergenic giving an irritation-free rest., Breathable and cool bamboo fabric that keeps you dry and cool throughout the night.Plushed with Generous Filling (1800g microfiber), Designer Envelope Style Closure, Breathable and Quality Materials, Hypoallergenic giving an irritation-free rest., Breathable and…See moreColourGreySizeBody (20x55)Fill materialMicrofiber PolyesterShapeRectangularAge range (description)Youth and AdultRecommended uses for productReading, SleepingUse forWhole BodyMaterialSilk, Polyester Casing, Rayon from bamboo, 1800g Microfiber filling",
        "sp_description": "BrandAnnamite RangeSpecial featurePlushed with Generous Filling (1800g microfiber), Designer Envelope Style Closure, Breathable and Quality Materials, Hypoallergenic giving an irritation-free rest., Breathable and cool bamboo fabric that keeps you dry and cool throughout the night.Plushed with Generous Filling (1800g microfiber), Designer Envelope Style Closure, Breathable and Quality Materials, Hypoallergenic giving an irritation-free rest., Breathable and…See moreColourGreySizeBody (20x55)Fill materialMicrofiber PolyesterShapeRectangularAge range (description)Youth and AdultRecommended uses for productReading, SleepingUse forWhole BodyMaterialSilk, Polyester Casing, Rayon from bamboo, 1800g Microfiber filling",
        "sp_about": "About this item    ✨ Breathable and Cool: Our body pillow features a breathable bamboo fabric that keeps you cool throughout the night. Say goodbye to sweaty nights and hello to blissful slumber.    ✨ Premium Quality Materials: We take pride in using only the finest materials. The 100% polyester microfiber casing ensures longevity, while the 1800g microfiber filling maintains its plushness. Rest assured, this pillow will be your trusted companion for years to come.    ✨ Designer Envelope-Style Pillowcase: The included pillowcase is not just any cover—it’s a stylish designer envelope-style case. Choose from three elegant colors: black, white, or grey. The silky soft bamboo fabric adds a touch of luxury to your sleep sanctuary.    ✨ Expert Craftsmanship: Our body pillow undergoes professional manufacturing, adhering to strict quality standards. From stitching to finishing touches, every detail is meticulously executed.    ✨ Care Instructions : To keep your Annamite Range Silky Soft Bamboo Body Pillow in pristine condition, follow these simple steps: Machine Washable: When needed, toss the pillowcase into the washing machine on a gentle cycle. Tumble Dry Low: For quick drying, tumble dry the pillowcase on low heat. Fluff Regularly: Give your pillow a gentle shake to maintain its loftiness.    🌙✨ Invest in quality sleep with the Annamite Range Silky Soft Bamboo Body Pillow. Experience the difference that thoughtful design and premium materials make.    🌙✨ Package includes one fluffy body pillow, one silky bamboo body pillowcase and one pillow bag.    \n ›  See more product details",
        "Buy Link": "https://www.amazon.ca/sspa/click?ie=UTF8&spc=MTo3Mjk1OTU4MDcwNTkyMjA3OjE3MzMzMDM3NTg6c3BfYnRmOjMwMDM3MTk0OTY1NjAwMjo6MDo6&url=%2FAnnamite-Range-Hypoallergenic-Envelope-Style-Pillowcase%2Fdp%2FB0CWZFVY7D%2Fref%3Dsr_1_60_sspa%3Fdib%3DeyJ2IjoiMSJ9.q9kApQiMqmEEI1nTAzCebXrSHTbJHBwHS0tO_v-UHEwNUvgnzRfXeR7GNcXBf_NjssAYQgqK69I0bvlIB0ZK4xmbhl8hqxCfXsGhJzbILzjwOzuDAoXJO8UomXmSOoctcx7CDPcknlfDcif0ackJ_SY4rYBuPOte_ukV673Ysl5RqHCBWRjFdsNObxSjoedXUaTyI9GIGKs3U6vGu-7lm9hnJ0qhK_LwRIt7uZF3WARHiVyqMDZMi_2_36FFQCTtuYAFgClJbKaDflSY-xkVH40wFg-H4zFIAcbGgyit8_k.DFiun5ZfPuE5_6U89Hugoj01cAA__mbSc8i1WxnDj4g%26dib_tag%3Dse%26keywords%3DPregnancy%2BPillows%26qid%3D1733303758%26sr%3D8-60-spons%26sp_csd%3Dd2lkZ2V0TmFtZT1zcF9idGY%26psc%3D1",
        "Reviews": "",
        "Review Count": 18,
        "Rating": "4.5",
        "Price": "$49.98",
        "Source": "https://www.amazon.ca/sspa/click?ie=UTF8&spc=MTo3Mjk1OTU4MDcwNTkyMjA3OjE3MzMzMDM3NTg6c3BfYnRmOjMwMDM3MTk0OTY1NjAwMjo6MDo6&url=%2FAnnamite-Range-Hypoallergenic-Envelope-Style-Pillowcase%2Fdp%2FB0CWZFVY7D%2Fref%3Dsr_1_60_sspa%3Fdib%3DeyJ2IjoiMSJ9.q9kApQiMqmEEI1nTAzCebXrSHTbJHBwHS0tO_v-UHEwNUvgnzRfXeR7GNcXBf_NjssAYQgqK69I0bvlIB0ZK4xmbhl8hqxCfXsGhJzbILzjwOzuDAoXJO8UomXmSOoctcx7CDPcknlfDcif0ackJ_SY4rYBuPOte_ukV673Ysl5RqHCBWRjFdsNObxSjoedXUaTyI9GIGKs3U6vGu-7lm9hnJ0qhK_LwRIt7uZF3WARHiVyqMDZMi_2_36FFQCTtuYAFgClJbKaDflSY-xkVH40wFg-H4zFIAcbGgyit8_k.DFiun5ZfPuE5_6U89Hugoj01cAA__mbSc8i1WxnDj4g%26dib_tag%3Dse%26keywords%3DPregnancy%2BPillows%26qid%3D1733303758%26sr%3D8-60-spons%26sp_csd%3Dd2lkZ2V0TmFtZT1zcF9idGY%26psc%3D1",
        "Category": "Pregnancy Pillows",
        "batch_id": "6500cc11-781b-4c6a-8794-bfe488464b1b"
    },
    {
        "Product Name": "Couple Pillow for Cuddling | Memory Foam Couple Pillow with Arm Hole | Anti Hand Pressure Reach Through Cuddle Pillows for Sleeping | Snuggle Cuddling Pillow | Valentine's Romantic Gifts",
        "Images": [
            {
                "url": "https://m.media-amazon.com/images/I/71gpXK0qKlL._AC_SL1500_.jpg"
            }
        ],
        "Image Link": "https://m.media-amazon.com/images/I/71gpXK0qKlL._AC_SL1500_.jpg",
        "sp_other_details": "Product information Collapse all Expand all Measurements Item Dimensions L x W 68.6L x 30.5W Centimetres Style Shape Rectangular Pattern Solid Colour White Materials & Care Material Features Breathable Fabric Type Memory Foam Material Memory Foam Product Care Instructions Machine Wash Fill Material Memory Foam Feedback Would you like to tell us about a lower price? Couple Pillow for Cuddling | Memory Foam Couple Pillow with Arm Hole | Anti Hand Pressure Reach Through Cuddle Pillows for Sleeping | Snuggle Cuddling Pillow | Valentine's Romantic Gifts Share: Found a lower price? Let us know. Although we can't match every price reported, we'll use your feedback to ensure that our prices remain competitive. Where did you see a lower price? Fields with an asterisk * are required Price Availability Website (Online) URL *: Price incl. GST (CAD) *: Shipping cost (CAD): Date of the price (MM/DD/YYYY): 01 02 03 04 05 06 07 08 09 10 11 12 / 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 / Store (Offline) Store name *: Enter the store name where you found this product Enter the store name where you found this product City *: Province: Please select province Please select province Price incl. GST (CAD) *: Date of the price (MM/DD/YYYY): 01 02 03 04 05 06 07 08 09 10 11 12 / 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 / Submit Feedback Please sign in to provide feedback. User guide Recommended Uses For Product Side Sleeper, Couples Cuddling Item details Manufacturer Lion Island ASIN B0BQMC9868 Customer Reviews 1.8 1.8 out of 5 stars 3 ratings 1.8 out of 5 stars Best Sellers Rank #759,809 in Home (See Top 100 in Home) #1,346 in Specialty Medical Pillows Model Name Cuddle 1 ProductIDType 718157832770 Brand Milki Age Range Description Adult Number of Pieces 1 Features & Specs Item Firmness Description Soft Special Features breathable, adjustable",
        "Description": "MILKI CUDDLE PILLOWS FOR COUPLESAre you looking for couples pillow with arm hole so you can cuddle and sleep together?Or are you looking for side arm sleeper pillow made of memory foam?Meet theMilki Couples Pillows for Sleeping, which are 100% made of breathable memory foam and designed with ergonomic arch shape for utmost comfort. Add this side arm sleeper pillow with arm tunnel to your bedroom to ignite passion, and get rid of that painful numbness when side sleeping, together or alone.►SUPPORTIVE, YET SOFTThe Milki couple pillows for him and her use memory foam which gives you the optimal support and adjustable softness. The spooning pillow easily adjusts to your head shape, neck, and back, while providing enough room to place your arm in a comfortable position that will stop the numbness caused from regular pillows.►REDISCOVER YOUR PASSION & CLOSSENESSSleeping on the opposite side separated from your partner may have a worrying effect on your passion and closeness. This pillow with arm hole for cuddling is here to make sleeping while being hugged a reality and enable hours and hours of cuddling.►COMES WITH PILLOWCASEEach Milki arm hole pillow side sleeper product comes with a removable pillowcase that is machine washable.►LOVELY GIFT IDEASurprise him or her with this couples pillow for arm numbness and get sincere gratitude for the thoughtful gift. Makes a perfect gift for Valentine's, anniversaries, birthdays, and more.Whether you are a side sleeper and need a comfortable hand pillow with an under pillow arm hole, or you need a couples pillow for cuddling, this is the ideal option.ClickAdd to Cartand get your Milki side sleeper pillow with arm hole NOW!",
        "sp_description": "MILKI CUDDLE PILLOWS FOR COUPLESAre you looking for couples pillow with arm hole so you can cuddle and sleep together?Or are you looking for side arm sleeper pillow made of memory foam?Meet theMilki Couples Pillows for Sleeping, which are 100% made of breathable memory foam and designed with ergonomic arch shape for utmost comfort. Add this side arm sleeper pillow with arm tunnel to your bedroom to ignite passion, and get rid of that painful numbness when side sleeping, together or alone.►SUPPORTIVE, YET SOFTThe Milki couple pillows for him and her use memory foam which gives you the optimal support and adjustable softness. The spooning pillow easily adjusts to your head shape, neck, and back, while providing enough room to place your arm in a comfortable position that will stop the numbness caused from regular pillows.►REDISCOVER YOUR PASSION & CLOSSENESSSleeping on the opposite side separated from your partner may have a worrying effect on your passion and closeness. This pillow with arm hole for cuddling is here to make sleeping while being hugged a reality and enable hours and hours of cuddling.►COMES WITH PILLOWCASEEach Milki arm hole pillow side sleeper product comes with a removable pillowcase that is machine washable.►LOVELY GIFT IDEASurprise him or her with this couples pillow for arm numbness and get sincere gratitude for the thoughtful gift. Makes a perfect gift for Valentine's, anniversaries, birthdays, and more.Whether you are a side sleeper and need a comfortable hand pillow with an under pillow arm hole, or you need a couples pillow for cuddling, this is the ideal option.ClickAdd to Cartand get your Milki side sleeper pillow with arm hole NOW!",
        "sp_about": "About this item    SIDE SLEEPER ARM PILLOW WITH OPTIMAL SOFTNESS: Are you a side sleeper? Well, you are set to enjoy deeper and longer sleeps with the Milki Memory Foam Pillow with Arm Hole for Side Sleeping. Designed with the optimal height, the memory foam adjusts to your head, shoulders, and favorite height so you can forget about arm numbness or discomfort.    CUDDLE PILLOW WITH ARM HOLE: The long design of this reach through pillow makes it a perfect memory foam couples cuddle pillow as well. Keep romance at a high and forget about the frustration of having to sleep on separate sides because of arm numbness or shoulder pain. Just slide the arm in the arm hole and enjoy a full night's sleep together on the couples pillow.    BREATHABLE AND ERGONOMIC DESIGN: The 100% memory foam material is breathable and ergonomic as it adjusts to your body shape to relieve pressure, tension, and pain. Even more, this arch-shaped pillow for arm under pillow restores its original shape by the time you are set to use it again.    EASY MAINTENANCE: The snuggle pillow comes with a removable pillowcase that can be machine washed for easy care. Meaning this can be your arm pillow for side sleepers for years to come. Even more, the memory foam doesn't lose its shape and resistance which aids in the pillow’s longevity.    LOVELY GIFT IDEA: This pillow for side sleepers with arm hole is the perfect gift idea for friends and family. Also, since it's long enough to be used as an arm pillow for cuddling it's a thoughtful valentine's gifts idea for her or him, or boyfriend snuggle pillow. Additionally, the cuddling pillow for couples is the ideal romantic gift to stay connected with your loved one and maintain passion and keep the feelings burning even while sleeping.    \n ›  See more product details",
        "Buy Link": "https://www.amazon.ca/sspa/click?ie=UTF8&spc=MTo2MDgwNzY4MTA4ODE3MTQ4OjE3MzMzMDM3MDA6c3BfYnRmOjIwMDE0MTA5ODU5ODY5ODo6MDo6&url=%2FCuddling-Pressure-Sleeping-Valentines-Romantic%2Fdp%2FB0BQMC9868%2Fref%3Dsr_1_57_sspa%3Fdib%3DeyJ2IjoiMSJ9.q9kApQiMqmEEI1nTAzCebXrSHTbJHBwHS0tO_v-UHEwNUvgnzRfXeR7GNcXBf_NjssAYQgqK69I0bvlIB0ZK4xmbhl8hqxCfXsGhJzbILzjwOzuDAoXJO8UomXmSOoctcx7CDPcknlfDcif0ackJ_SY4rYBuPOte_ukV673Ysl5RqHCBWRjFdsNObxSjoedXUaTyI9GIGKs3U6vGu-7lm9hnJ0qhK_LwRIt7uZF3WARHiVyqMDZMi_2_36FFQCTtuYAFgClJbKaDflSY-xkVH40wFg-H4zFIAcbGgyit8_k.DFiun5ZfPuE5_6U89Hugoj01cAA__mbSc8i1WxnDj4g%26dib_tag%3Dse%26keywords%3DPregnancy%2BPillows%26qid%3D1733303700%26sr%3D8-57-spons%26sp_csd%3Dd2lkZ2V0TmFtZT1zcF9idGY%26psc%3D1",
        "Reviews": "",
        "Review Count": 3,
        "Rating": "1.8",
        "Price": "$110.99",
        "Source": "https://www.amazon.ca/sspa/click?ie=UTF8&spc=MTo2MDgwNzY4MTA4ODE3MTQ4OjE3MzMzMDM3MDA6c3BfYnRmOjIwMDE0MTA5ODU5ODY5ODo6MDo6&url=%2FCuddling-Pressure-Sleeping-Valentines-Romantic%2Fdp%2FB0BQMC9868%2Fref%3Dsr_1_57_sspa%3Fdib%3DeyJ2IjoiMSJ9.q9kApQiMqmEEI1nTAzCebXrSHTbJHBwHS0tO_v-UHEwNUvgnzRfXeR7GNcXBf_NjssAYQgqK69I0bvlIB0ZK4xmbhl8hqxCfXsGhJzbILzjwOzuDAoXJO8UomXmSOoctcx7CDPcknlfDcif0ackJ_SY4rYBuPOte_ukV673Ysl5RqHCBWRjFdsNObxSjoedXUaTyI9GIGKs3U6vGu-7lm9hnJ0qhK_LwRIt7uZF3WARHiVyqMDZMi_2_36FFQCTtuYAFgClJbKaDflSY-xkVH40wFg-H4zFIAcbGgyit8_k.DFiun5ZfPuE5_6U89Hugoj01cAA__mbSc8i1WxnDj4g%26dib_tag%3Dse%26keywords%3DPregnancy%2BPillows%26qid%3D1733303700%26sr%3D8-57-spons%26sp_csd%3Dd2lkZ2V0TmFtZT1zcF9idGY%26psc%3D1",
        "Category": "Pregnancy Pillows",
        "batch_id": "6500cc11-781b-4c6a-8794-bfe488464b1b"
    }
]



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

save_to_airtable(a,"Pregnancy Pillows","bf1f0181-4d2a-4426-9ee6-dfa448440e34","YBC CA")