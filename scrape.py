from urllib.parse import urlparse
import requests
import xml.etree.ElementTree as ET
import re
from bs4 import BeautifulSoup
import json
import spacy
import openai
from google.cloud import bigquery
from datetime import datetime
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import streamlit as st


class QueryModel(BaseModel):
    query: str

app = FastAPI()

# Load the spaCy model
nlp = spacy.load('en_core_web_md')
client = bigquery.Client()

load_dotenv()
open_ai_key = os.getenv("OPENAI_API_KEY")
client = bigquery.Client()

load_dotenv()
open_ai_key = os.getenv("OPENAI_API_KEY")

def save_to_json(data, filename):
    """Save the scraped data to a JSON file."""
    try:
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)
    except IOError as e:
        print(f"Error saving data to {filename}: {e}")


def fetch_robots_txt(url):
    try:
        response = requests.get(url + "/robots.txt")
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching robots.txt: {e}")
        return None

def find_sitemap_url(robots_txt_content):
    for line in robots_txt_content.splitlines():
        if line.startswith('Sitemap:'):
            return line.split(': ')[1].strip()
    return None

def fetch_xml_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        print(f"Error fetching XML content from {url}: {e}")
        return None

def filter_urls(urls, exclude_pattern):
    """Filter URLs based on the exclude pattern."""
    filtered_urls = [url for url in urls if not re.search(exclude_pattern, url)]
    return filtered_urls

def scrape_page_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()

        # Parse the content with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract text from paragraph tags
        paragraphs = soup.find_all('p')
        text_content = [para.get_text() for para in paragraphs]

        return text_content
    except requests.RequestException as e:
        print(f"Error fetching page {url}: {e}")
        return None
    
def parse_sitemap(sitemap_content, include_pattern=None, exclude_pattern=None):
    try:
        root = ET.fromstring(sitemap_content)
        namespaces = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [url.text for url in root.findall('ns:url/ns:loc', namespaces)]

        if include_pattern:
            filtered_urls = [url for url in urls if re.search(include_pattern, url)]
        elif exclude_pattern:
            filtered_urls = [url for url in urls if not re.search(exclude_pattern, url)]
        else:
            filtered_urls = urls  # No filtering, return all URLs

        return filtered_urls
    except ET.ParseError as e:
            print(f"Error parsing XML content: {e}")
            return []



def process_sitemap(website_name, sitemap_url, main_pages_pattern=None, exclude_pattern=None):
def process_sitemap(website_name, sitemap_url, main_pages_pattern=None, exclude_pattern=None):
    sitemap_content = fetch_xml_content(sitemap_url)
    scraped_data = {}
    if sitemap_content:

        root = ET.fromstring(sitemap_content)
        if root.tag.endswith('sitemapindex'):
            nested_sitemaps = [loc.text for loc in root.findall('ns:sitemap/ns:loc', {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'})]
            for nested_sitemap_url in nested_sitemaps:
                nested_sitemap_content = fetch_xml_content(nested_sitemap_url)
                if nested_sitemap_content:
                    if main_pages_pattern:
                        urls = parse_sitemap(nested_sitemap_content, main_pages_pattern)
                    else:
                        urls = parse_sitemap(nested_sitemap_content, exclude_pattern)
                    
                    
                    i = 0

                    for url in urls:
                        print(url)
                        content = scrape_page_content(url)
                        print(content)
                        if content:
                            data = {
                                "url": url,
                                "website_name": website_name,  # Replace with actual title extraction logic
                                "content": " ".join(content),
                                "date_added": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                            }
                            insert_chunks_into_bigquery(website_name,url,data)
                        # i += 1
                        # if(i>=5):
                        #     break

        elif root.tag.endswith('urlset'):
            if main_pages_pattern:
                urls = parse_sitemap(sitemap_content, main_pages_pattern)
            else:
                urls = parse_sitemap(sitemap_content, exclude_pattern)

            i = 0


            i = 0

            for url in urls:
                print(url)
                content = scrape_page_content(url)
                if content:
                    data = {
                        "url": url,
                        "website_name": website_name,  # Replace with actual title extraction logic
                        "content": " ".join(content),
                        "date_added": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    }
                    insert_chunks_into_bigquery(website_name,url,data)
                # i += 1
                # if(i>=5):
                #     break

    return scraped_data

def extract_key_terms(query):
    doc = nlp(query)

    # Exclude common interrogative words and other common words
    exclude_words = {'who', 'what', 'where', 'when', 'why', 'how', 'is', 'are', 'was', 'were', 'be', 'been', 'being','content','it'}

    # Extract noun chunks and key adjectives, excluding common question and stop words
    key_terms = [chunk.text for chunk in doc.noun_chunks if chunk.text.lower() not in exclude_words]
    key_terms += [token.text for token in doc if token.pos_ == 'ADJ' and token.text.lower() not in exclude_words and not token.is_stop]

    return key_terms

def extract_key_terms2(query):
    doc = nlp(query)

    # Named Entity Recognition for entities
    entities = set(ent.text.lower() for ent in doc.ents)

    # Extracting unique nouns, verbs, and adjectives
    keywords = set(token.text.lower() for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ'])

    # Combine and filter unique keywords
    combined_keywords = entities.union(keywords)

    return list(combined_keywords)

def process_query2(query, data):
    # Extract key terms from the query - this can be as simple or complex as you need
    key_terms = extract_key_terms(query)  # You'll need to implement this function
    key_terms = [term.lower() for term in key_terms]

    print("key_terms: ", key_terms)
    print("query: ", query)
    # print("key terms: ", key_terms)
    # Search for relevant content
    relevant_content = []
    for url, content in data.items():
        content = " ".join(content)  # Flatten content if it's a list
        if any(term in content for term in key_terms):
            relevant_content.append((url, content))

    return relevant_content

def process_query(query, data):
    doc_query = nlp(query)

    relevant_content = []
    for url, content_list in data.items():
        content = " ".join(content_list)
        doc_content = nlp(content)

        for sentence in doc_content.sents:
            similarity = doc_query.similarity(sentence)
            if similarity > 0.7:  # Adjusted threshold for higher accuracy
                relevant_content.append((url, sentence.text, similarity))

    # Sort by similarity score and return top results
    top_results = sorted(relevant_content, key=lambda x: x[2], reverse=True)[:10]  # Increase the number of results
    return top_results


def get_page_title(url):
    """
    Extract a human-readable page title from a URL.
    :param url: String URL.
    :return: String representing a human-readable page title.
    """
    path = urlparse(url).path
    # Extract the last part of the URL path and replace hyphens with spaces
    return path.split('/')[-1].replace('-', ' ').replace('.html', '').capitalize()

def summarize_content(data, query, max_sentences=5, max_sentence_length=10):
    """
    Summarize content by extracting key sentences containing query terms.
    :param data: List of tuples, each tuple contains a URL and a string of content.
    :param query: Query string based on which content is summarized.
    :param max_sentences: Maximum number of sentences to include in the summary.
    :param max_sentence_length: Maximum number of words in each sentence.
    :return: List of tuples with page title and summarized string.
    """
    summarized_results = []
    key_terms = extract_key_terms2(query)

    for url, content, similarity in data:
        page_title = get_page_title(url)
        sentences = [sentence.strip() for sentence in re.split(r'[.!?]', content) if sentence]
        short_sentences = [s for s in sentences if len(s.split()) <= max_sentence_length]

        # Rank sentences based on the occurrence of key terms
        ranked_sentences = sorted(short_sentences, key=lambda s: sum(term.lower() in s.lower() for term in key_terms), reverse=True)

        # Select top sentences up to the max_sentences limit
        summary_sentences = ranked_sentences[:max_sentences]
        unique_sentences = list(dict.fromkeys(summary_sentences))  # Remove duplicates
        summary = ' '.join(unique_sentences).strip()

        summarized_results.append((page_title, summary))

    return summarized_results

def generate_answer_with_gpt(query, relevant_content):
    try:
        openai.api_key = open_ai_key
        openai.api_key = open_ai_key

        summarized_content = summarize_content(relevant_content,query)
        # print ("summary: ", summarized_content)
        if summarized_content:
        
            # Formulate the prompt for GPT
            prompt = f"I want you to find an answer with this question : {query} \n Find the answer in this contents: {summarized_content}\n\n"

            # Call the OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-4-0613",  # You can choose different engines as needed
                messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            )

            print(response)
            # Extracting the response text
            last_message = response['choices'][0]['message']['content'] if response['choices'] else "No response generated."
            return last_message.strip()
        else:
            return "No relevant content found for the query."

    except openai.error.OpenAIError as e:
            print(f"Error in OpenAI API call: {e}")
            return "Sorry, I couldn't process the request."
    
def process_query_bigquery2(query):
    key_terms = extract_key_terms(query)
    
    
    # Create a series of LIKE clauses for each key term
    like_clauses = " OR ".join([f"content LIKE '%{term}%'" for term in key_terms])

    query_sql = f"""
    SELECT url, content
    FROM `qanda-website-ai.websites.website_content`
    WHERE {like_clauses} LIMIT 20
    """

    print(query_sql)  # For debugging
    query_job = client.query(query_sql)  # Make an API request.

    relevant_content = []
    for row in query_job:
        relevant_content.append((row.url, row.content))

    return relevant_content

def contains_discount_phrases(content):
    # Define discount-related phrases and regex patterns
    discount_keywords = ["discount", "promo code", "coupon", "sale"]
    discount_regex = r"\b\d+% off\b|\b(save|get) \$.+\b"

    content_lower = content.lower()
    # Check for presence of discount keywords
    if any(keyword in content_lower for keyword in discount_keywords):
        return True

    # Check for regex pattern matches
    if re.search(discount_regex, content_lower):
        return True

    return False


def process_query_bigquery(query):
    key_terms = extract_key_terms2(query)
    print("Key terms:", key_terms)

    like_clauses = " OR ".join([f"LOWER(content) LIKE '%{term}%'" for term in key_terms])

    query_sql = f"""
    SELECT url, content
    FROM `qanda-website-ai.websites.website_content`
    WHERE {like_clauses} LIMIT 20
    """
    print(query_sql)  # For debugging
    client = bigquery.Client()
    query_job = client.query(query_sql)  # Make an API request.

    doc_query = nlp(query)
    relevant_content = []

    for row in query_job:
        # Pre-filtering: Split content into sentences and check for key term presence
        if contains_discount_phrases(row.content):
            relevant_content.append((row.url, row.content, 1.0))  # Assigning high similarity for direct matches
        else:
            sentences = [sentence.strip() for sentence in re.split(r'[.!?]', row.content) if sentence]
            if any(term in " ".join(sentences).lower() for term in key_terms):
                doc_content = nlp(" ".join(sentences))
                similarity = doc_query.similarity(doc_content)
                if similarity > 0.55:  # Adjust the threshold as needed
                    relevant_content.append((row.url, row.content, similarity))

    return relevant_content

def questions(q):

    user_query = q  # Replace this with actual user input
    relevant_content = process_query_bigquery(user_query)
    print("content: ", relevant_content)
    # Use NLP/AI to generate an answer from relevant_content
    # answer = generate_answer(relevant_content, user_query)  # This is a placeholder for the AI integration

    if relevant_content:
        answer = generate_answer_with_gpt(user_query, relevant_content)
        print("Question: ",user_query)
        print("")
        print("Answer:", answer)

        return answer
    else:
        print("No relevant content found for the query.")

def insert_into_bigquery(data):
    table_id = "qanda-website-ai.websites.website_content"  # Replace with your actual table ID
    table = client.get_table(table_id)
    rows_to_insert = [data]

    errors = client.insert_rows_json(table, rows_to_insert)
    if errors == []:
        print("New rows have been added.")
    else:
        print("Encountered errors while inserting rows: {}".format(errors))

def insert_chunks_into_bigquery(website_name, url, content, batch_size=500, max_paragraph_length=500):
    table_id = "qanda-website-ai.websites.website_content"  # Replace with your actual table ID
    client = bigquery.Client()
    table = client.get_table(table_id)

    # Ensure content is a string
    if content is None:
        print(f"No content to insert for {url}.")
        return
    if not isinstance(content, str):
        content = str(content)

    # Function to further split large paragraphs
    def split_large_paragraph(paragraph, max_length):
        # Split by sentences if possible
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        if len(paragraph) <= max_length or not sentences:
            return [paragraph]

        small_chunks = []
        current_chunk = sentences[0]
        for sentence in sentences[1:]:
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += " " + sentence
            else:
                small_chunks.append(current_chunk.strip())
                current_chunk = sentence
        if current_chunk:
            small_chunks.append(current_chunk.strip())

        return small_chunks

    # Split the content into chunks
    paragraphs = re.split(r'\n\s*\n', content)
    all_chunks = []
    for paragraph in paragraphs:
        if len(paragraph.strip()) <= max_paragraph_length:
            all_chunks.append(paragraph.strip())
        else:
            all_chunks.extend(split_large_paragraph(paragraph, max_paragraph_length))

    # Prepare rows for batch insert
    rows_to_insert = [{
        "website_name": website_name,
        "url": url,
        "content": chunk,
        "date_added": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    } for chunk in all_chunks if chunk]

    # Batch insert chunks
    for i in range(0, len(rows_to_insert), batch_size):
        batch = rows_to_insert[i:i + batch_size]
        errors = client.insert_rows_json(table, batch)
        if not errors:
            print(f"Batch of rows has been added for {url}.")
        else:
            print(f"Encountered errors while inserting batch for {url}: {errors}")


def begin_scraping(website_name, website_url):
    website_name = "bluemercury"
    website_url = "https://bluemercury.com"  # Replace with the target website URL
    exclude_pattern = r'/(product|products|collection|collections|staging)/'  # Regular expression to exclude URLs
    main_pages_pattern = r'/(about|faq|contact|home|pages)/'

    # Scrape the homepage content directly
    # homepage_content = scrape_page_content(website_url)
    # scraped_data = {website_url: homepage_content} if homepage_content else {}

    # Scrape the homepage content directly
    homepage_content = scrape_page_content(website_url)
    if homepage_content:
        combined_content = " ".join(homepage_content)
        homepage_data = {
            "url": website_url,
            "website_name": website_name,
            "content": " ".join(homepage_content),
            "date_added": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        }
        insert_chunks_into_bigquery(website_name,website_url,combined_content)

    # Process sitemap
    robots_txt_content = fetch_robots_txt(website_url)
    if robots_txt_content:
        sitemap_url = find_sitemap_url(robots_txt_content)
        if sitemap_url:
            process_sitemap(website_name, sitemap_url, main_pages_pattern, exclude_pattern)
        else:
            print("Sitemap URL not found in robots.txt")
    else:
        print("Failed to fetch robots.txt")

# if __name__ == "__main__":
#     questions()

@app.post("/scrape")
async def scrape_endpoint(url: str):
    # Your scraping logic here
    result = await begin_scraping(url)
    return {"result": result}

@app.post("/process_query")
def process_query_endpoint(query_model: QueryModel):
# Your query processing logic here
    query = query_model.query
    response = questions(query)
    return {"response": response}

# Streamlit app starts here
st.title('Question & Answer AI')

query = st.text_input("Enter your query:")

if query:

    try:
        st.write("Processing your query...")
        results = questions(query)
        if results:
            st.write(f"Answer : {results}")
        
        else:
            st.write("No relevant content found.")
    except Exception as e:
            st.error(f"An error occurred: {e}")