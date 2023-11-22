import requests
import xml.etree.ElementTree as ET
import re
from bs4 import BeautifulSoup
import json
import spacy
import openai

# Load the spaCy model
nlp = spacy.load('en_core_web_md')

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



def process_sitemap(sitemap_url, main_pages_pattern=None, exclude_pattern=None):
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
                            scraped_data[url] = content
                        # i += 1
                        # if(i>=5):
                        #     break

        elif root.tag.endswith('urlset'):
            if main_pages_pattern:
                urls = parse_sitemap(sitemap_content, main_pages_pattern)
            else:
                urls = parse_sitemap(sitemap_content, exclude_pattern)
            for url in urls:
                print(url)
                content = scrape_page_content(url)
                if content:
                    scraped_data[url] = content
                # i += 1
                # if(i>=5):
                #     break

    return scraped_data

def extract_key_terms(query):
    doc = nlp(query)

    # Exclude common interrogative words
    exclude_words = {'who', 'what', 'where', 'when', 'why', 'how'}

    # Extract noun chunks and key adjectives, excluding common question words
    key_terms = [chunk.text for chunk in doc.noun_chunks if chunk.text.lower() not in exclude_words]
    key_terms += [token.text for token in doc if token.pos_ == 'ADJ' and token.text.lower() not in exclude_words]

    return key_terms

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
            if similarity > 0.5:  # Threshold for similarity, adjust as needed
                relevant_content.append((url, sentence.text, similarity))

    # Sort by similarity score and return top results
    top_results = sorted(relevant_content, key=lambda x: x[2], reverse=True)[:5]  # Adjust the number as needed
    return top_results


def generate_answer_with_gpt(query, relevant_content):
    try:
        openai.api_key = "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"

        # Combine the relevant content into a single string
        combined_content = " ".join([content for _, content in relevant_content])
        

        # Formulate the prompt for GPT
        prompt = f"I want you to find an answer with this question : {query} \n Find the answer in this contents: {combined_content}\n\n"

        # Call the OpenAI API
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct-0914",  # You can choose different engines as needed
            prompt=prompt,
            max_tokens=500  # Adjust as needed
        )

        return response.choices[0].text.strip()

    except openai.error.OpenAIError as e:
            print(f"Error in OpenAI API call: {e}")
            return "Sorry, I couldn't process the request."

def questions():
    # print(scrape_page_content("https://bluemercury.com"))
    # Load scraped data from JSON
    with open("scraped_data.json", "r") as file:
        scraped_data = json.load(file)

    # Example query
    user_query = "Who is the target market for this store?"  # Replace this with actual user input
    relevant_content = process_query(user_query, scraped_data)
    print("content: ", relevant_content)
    # Use NLP/AI to generate an answer from relevant_content
    # answer = generate_answer(relevant_content, user_query)  # This is a placeholder for the AI integration

    if relevant_content:
        answer = generate_answer_with_gpt(user_query, relevant_content)
        print("Question: ",user_query)
        print("")
        print("Answer:", answer)
    else:
        print("No relevant content found for the query.")
    

def main():
    website_url = "https://developer.weweb.io/"  # Replace with the target website URL
    exclude_pattern = r'/(product|products|collection|collections)/'# Regular expression to exclude URLs
    main_pages_pattern = r'/(about|faq|contact|home|pages)/'

    robots_txt_content = fetch_robots_txt(website_url)
    if robots_txt_content:
        sitemap_url = find_sitemap_url(robots_txt_content)
        if sitemap_url:
            scraped_data = process_sitemap(sitemap_url, main_pages_pattern, exclude_pattern)
            save_to_json(scraped_data, "scraped_data.json")
        else:
            print("Sitemap URL not found in robots.txt")
    else:
        print("Failed to fetch robots.txt")

if __name__ == "__main__":
    questions()
