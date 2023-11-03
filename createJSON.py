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

    # Remove trailing commas
    text = re.sub(r',\s*$', '', text)
    
    return text

def generate_unique_filename(batch_id, article_title):
    # Extract a substring from the batch_id starting at index 12 and ending at index 17
    batch_id_substring = batch_id[12:18]
    
    # Replace any spaces in the article_title with underscores
    article_title_cleaned = article_title.replace(" ", "_")
    
    # Create a unique filename
    unique_filename = f"{batch_id_substring}-{article_title_cleaned}.jpg"
    
    return unique_filename

def extract_section(title, text, max_length=None):
    if f"{title}:" in text:
        cleaned_text = text.replace(f"{title}:", "").strip()
    elif f"{title}" in text:
        cleaned_text = text.replace(f"{title}", "").strip()
    else:
        cleaned_text = text.strip()
    cleaned_text = clean_text(cleaned_text)
    if max_length is not None:
        lines = cleaned_text.split('\n')
        lines = [line for line in lines if len(line) <= max_length]
        cleaned_text = '\n'.join(lines)
    return cleaned_text

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
        review_section['BuyLink'] = product.get('Buy Link', '') + "|Buy Now From Amazon"
        price = product.get('Price', '')
        if price != "N/A":
            try:
                price = float(price)
                review_section['Price'] = "${:,.2f}".format(price)
            except ValueError:
                review_section['Price'] = price

        
        long_description = product.get('Long Description', '')
        
        sections = re.split(r'####\s*', long_description)

        for section in sections:
            if "Our Review" in section:
                review_section['ReviewOverview'] = extract_section("Our Review", section)
            elif "What We Like" in section:
                review_section['WhatWeLike'] = extract_section("What We Like", section)
            elif "What's in the Box" in section:
                review_section['WhatsInTheBox'] = extract_section("What's in the Box", section)
            elif "Additional Information" in section:
                review_section['UsefulProductInfo'] = extract_section("Additional Information", section,115)

                

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
