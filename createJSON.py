import json
import re


def clean_text(text):
    # Remove leading and trailing whitespaces
    text = text.strip()
    
    # Remove dashes at the beginning of each line
    text = re.sub(r'^-\s*', '', text, flags=re.MULTILINE)
    
    # Remove other unnecessary characters
    text = re.sub(r'\*', '', text)
    
    # Remove consecutive newlines and spaces surrounding them
    text = re.sub(r'\n\s*\n', '\n', text)
    
    return text

def generate_unique_filename(batch_id, article_title):
    # Extract a substring from the batch_id starting at index 12 and ending at index 17
    batch_id_substring = batch_id[12:18]
    
    # Replace any spaces in the article_title with underscores
    article_title_cleaned = article_title.replace(" ", "_")
    
    # Create a unique filename
    unique_filename = f"{batch_id_substring}-{article_title_cleaned}.jpg"
    
    return unique_filename


def transform_product_data(input_json_file, output_json_file):
   # Read the original JSON file
   # Load JSON data from input file-like object
    data = json.load(input_json_file)
    
    # Parse the JSON string in the 'Products' field
    products_json_str = data[0]['Products']
    if isinstance(products_json_str, str):
        parsed_data = json.loads(products_json_str)
    else:
        parsed_data = products_json_str

    # Check if parsed_data is a list with a single string element
    if isinstance(parsed_data, list) and len(parsed_data) == 1 and isinstance(parsed_data[0], str):
        parsed_data = json.loads(parsed_data[0])  # Parse the string again

    # Initialize the transformed data with the specific structure
    transformed_data = {
        'Introduction': '',
        'IntroductionSection': [],
        'ReviewHighlightTitle': '',
        'ReviewSection': [],
        'FaqHeader': 'Frequently Asked Questions',
        'FaqItems': []
    }

    # Mapping for the 'Introduction', 'IntroductionSection', and 'ReviewHighlightTitle'
    transformed_data['Introduction'] = data[0].get('Intro', '')
    transformed_data['IntroductionSection'] = [{
        'BestBuyTitle': 'Our Best Buy',
        'BestBuyDescription': data[0].get('Our Best Buy', '')
    }]
    transformed_data['ReviewHighlightTitle'] = data[0].get('Article Title', '')

    print("parse data")

    # Mapping for the "ReviewSection"
    for product in parsed_data:
        review_section = {}

        review_section['ProductTitle'] = product.get('Short Product Name', '')
        review_section['BuyLink'] = product.get('Buy Link', '')
        review_section['Price'] = product.get('Price', '')
        
        long_description = product.get('Long Description', '')
        sections = long_description.split('#### ')

        for section in sections:
            if "Our Review:" in section:
                review_section['ReviewOverview'] = clean_text(
                    section.replace("Our Review:", "").strip())
            elif "What We Like:" in section:
                review_section['WhatWeLike'] = clean_text(
                    section.replace("What We Like:", "").strip())
            elif "What's in the Box:" in section:
                review_section['WhatsInTheBox'] = clean_text(
                    section.replace("What's in the Box:", "").strip())
            elif "Additional Information:" in section:
                review_section['UsefulProductInfo'] = clean_text(
                    section.replace("Additional Information:", "").strip())
                

        unique_filename = generate_unique_filename(data[0].get('Article ID', ''), product.get('Short Product Name', ''))

        review_section['ReviewImage'] = [{"AssetAltText": product.get('Short Product Name', '') + " Review", "AssetFileName": unique_filename,
                                          "AssetImageUrl":  product.get('Image Link', ''), "AssetImageBase64": ""}]

        if product.get('Price') != "N/A":
            transformed_data['ReviewSection'].append(review_section)

        # Mapping for the "FAQSection"
    transformed_data['FaqHeader'] = "Frequently Asked Questions"
    transformed_data['FaqItems'] = []

    faqs = data[0].get('FAQs', '')
    faq_sections = faqs.split('#### ')[1:]

    for faq in faq_sections:
        faq_split = faq.split('\nA:')
        if len(faq_split) == 2:
            q, a = faq_split
            faq_dict = {
                'FaqTitle': clean_text(q.replace('Q:', '').strip()),
                'FaqAnswer': clean_text(a.strip())
            }
            transformed_data['FaqItems'].append(faq_dict)

    # Save to a new JSON file
    # with open(output_json_file, 'w') as json_file:
    json.dump(transformed_data, output_json_file, indent=4)


# Usage
# input_json_file = 'markdown.json'
# output_json_file = 'transformed_data.json'
# transform_product_data(input_json_file, output_json_file)
