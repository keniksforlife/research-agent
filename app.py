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
from langchain.schema import SystemMessage
from fastapi import FastAPI
import streamlit as st
import json
import logging



load_dotenv()
brwoserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")

logging.basicConfig(filename='my_log_file.txt',  # Name of the log file
                    level=logging.INFO,           # Set the logging level
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 1. Tool for search

def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query
    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)
    logging.info('SEARCH: %s', response.text)
    return response.text
# search("Most popular prams for newborns")

# 2. Tool for scraping
def scrape_website(objective: str, url: str):
    # scrape website, and also will summarize the content based on objective if the content is too large
    # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    try:
        print("Scraping website...")
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
            text = soup.get_text()
            print("CONTENTTTTTT:", text)

            if len(text) > 10000:
                output = detailed_summary(objective, text)
                logging.info('SCRAPE WEBSITE SUMMARY: %s', output)
                return output
            else:
                logging.info('SCRAPE WEBSITE text: %s', text)
                return text
        else:
            print(f"HTTP request failed with status code {response.status_code}")
    except requests.exceptions.ConnectionError as e:
        print(f"Failed to establish a new connection: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary in JSON Format of the following text for {objective}:
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

def detailed_summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    
    # Define multiple prompts
    prompts = {
        "summary": "Write a comprehensive summary focusing on features and benefits for {objective}: {text}",
        "what_we_like": "Enumerate 1-3 points highlighting what stands out for {objective}: {text}",
        "best_for": "Specify the type of user or situation this product is ideally suited for {objective}: {text}",
        "price": "Extract and provide the specific price or price range of the product for {objective}, if available in the content: {text}",
        "dimensions": "Provide any available dimensions like length, width, height for {objective}: {text}",
        "weight": "Extract and mention the weight of the product if it is available for {objective}: {text}",
        "image": "If an image URL of the product is present, please extract and return it for {objective}: {text}",
        "buy_link": "Extract and provide the direct link where this product can be purchased, if available in the content f{objective}: {text}"
}
    
    output = {}
    
    for info_type, prompt_template in prompts.items():
        map_prompt_template = PromptTemplate(
            template=prompt_template, input_variables=["text", "objective"])
        
        info_chain = load_summarize_chain(
            llm=llm,
            chain_type='map_reduce',
            map_prompt=map_prompt_template,
            combine_prompt=map_prompt_template,
            verbose=True
        )
        
        output[info_type] = info_chain.run(input_documents=docs, objective=objective)
        
    return json.dumps(output)  # Convert the output dictionary to JSON

# scrape_website("Chicco KeyFit 30","https://babygearessentials.com/chicco-keyfit-30/")

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
        logging.info('ScrapeWebsiteTool-BaseTool: %s', scrape_website(objective, url))
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
    # content1="""You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
    #         you do not make things up, you will try as hard as possible to gather facts & data to back up the research
            
    #         Please make sure you complete the objective above with the following rules:
    #         1/ You should do enough research to gather as much information as possible about the objective
    #         2/ If there are url of relevant links & articles, you will scrape it to gather more information
    #         3/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iteratins
    #         4/ You should not make things up, you should only write facts & data that you have gathered
    #         5/ In the final output, You should include all reference data & links to back up your research in json format; You should include all reference data & links to back up your research in json format
    #         6/ In the final output, You should include all reference data & links to back up your research in json format; You should include all reference data & links to back up your researc in json format"""
    content="""You are a world-class researcher, who can do detailed research on any topic and produce facts-based results; you do not make things up, you will try as hard as possible to gather facts & data to back up the research. You should research for top 10 best/most popular products base on the objective

            Please complete the following research objective while adhering to these rules:
            1/ You should do enough research to gather as much information as possible about the objective.
            2/ If there are URLs of relevant links & articles, please include them.
            3/ After gathering data, think, "Is there anything new I should search for based on the data I collected to increase research quality?" If the answer is yes, continue; But don't do this more than 3 iterations.
            4/ You should not make things up; you should only write facts & data that you have gathered.
            5/ In the final output, please return the research findings in a structured JSON format. The JSON should include a "research_summary", an array of "items" with details, and a "sources" array with reference links.
            6/ In the final output, please return the research findings in a structured JSON format. The JSON should include a "research_summary", an array of "items" with details, and a "sources" array with reference links.
            7/ In the final output, please return the research findings in a structured JSON format. The JSON should include a "research_summary", an array of "items" with details, and a "sources" array with reference links.
            8/ For example, if the research topic is 'Best Baby Car Seats for 2023', the JSON should look something like this:
            8/ {"research_summary":"Summary of the researçch...","items":[{"name":"Product Name","description":"Product Description","source":"URL Source","what_we_like":"List 1-3 points about what makes this product stand out","best_for":"Describe the type of user or situation this product is best suited for","price":"$0.00","dimension":"Provide dimensions such as length, width, height","weight":"weight","image":"image url","buy_link":"Extract and provide the direct link where this product can be purchased"}],"sources":[{"title":"Source Title","link":"Source Link"}]}
            10/ Please make sure the JSON is well-formatted and valid. Only returned one set of research_summary to avoid duplicate.
            11/ Please make sure the JSON is well-formatted and valid. Only returned one set of research_summary to avoid duplicate.
            12/ Please make sure the JSON is well-formatted and valid. Only returned one set of research_summary to avoid duplicate."""
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
def main():
    st.set_page_config(page_title="AI research agent", page_icon=":bird:")

    st.header("AI research agent :bird:")
    query = st.text_input("Research goal")

    if query:
        st.write("Doing research for ", query)

        result = agent({"input": query})

        st.info(result['output'])


if __name__ == '__main__':
    main()


# 5. Set this as an API endpoint via FastAPI
app = FastAPI()


class Query(BaseModel):
    query: str


@app.post("/")
def researchAgent(query: Query):
    query = query.query
    content = agent({"input": query})
    actual_content = content['output']

    # Step 2: Parse the string into a Python dictionary
    # parsed_dict = json.loads(actual_content.replace('\n', '').replace('\\', ''))

    return remove_duplicate_json(actual_content)



def remove_duplicate_json(json_str):

    json_str = json_str.replace('}}\n{{', '}\n{"')

    # Split the string by the delimiter '}{'
    json_list = json_str.split("}\n{")
    
    # Create an empty set to store unique JSON objects
    unique_json_set = set()
    
    # Iterate through the list and add each unique JSON object to the set
    for json_obj in json_list:
        # Add curly braces back to each JSON object
        if not json_obj.startswith("{"):
            json_obj = "{" + json_obj
        if not json_obj.endswith("}"):
            json_obj = json_obj + "}"
        
        unique_json_set.add(json_obj)
    
    # Join the unique JSON objects back into a single string
    unique_json_str = "}\n{".join(unique_json_set)
    
    print (unique_json_str)
    return unique_json_str

def remove_duplicate_research_summary(json_str_list):
    unique_research_dict = {}
    
    for json_str in json_str_list:
        parsed_json = json.loads(json_str)
        research_summary = parsed_json.get("research_summary")
        
        if research_summary not in unique_research_dict:
            unique_research_dict[research_summary] = parsed_json

    # Convert the unique dictionaries back to JSON strings
    unique_json_list = [json.dumps(item, indent=4) for item in unique_research_dict.values()]
    
    return unique_json_list


myjson = "{\"research_summary\": \"After conducting extensive research on the best baby sleeping bags, here are the top 10 products that are highly recommended:\", \"items\": [{\"name\": \"Kyte Baby Sleep Bag\", \"description\": \"The Kyte Baby Sleep Bag is considered the best overall sleep sack. It is made of high-quality materials and provides a cozy and comfortable sleeping environment for babies.\", \"source\": \"https://www.forbes.com/sites/forbes-personal-shopper/article/best-sleep-sacks/?sh=24a9e90d6458\", \"what_we_like\": \"Made of high-quality materials, Cozy and comfortable\", \"best_for\": \"All babies\", \"price\": \"$18.00\", \"dimension\": \"\", \"weight\": \"\", \"image\": \"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRzLqCbeJXBtTvnbOyrJF29XPTkhaeGoi2r-At9jYiZLLLjo3jOc3kDxag&s\", \"buy_link\": \"\"}, {\"name\": \"Burt's Bees Baby Beekeeper Blanket\", \"description\": \"The Burt's Bees Baby Beekeeper Blanket is a value sleep sack that offers both comfort and affordability. It is made of 100% organic cotton and has a snug fit to keep babies secure during sleep.\", \"source\": \"https://www.forbes.com/sites/forbes-personal-shopper/article/best-sleep-sacks/?sh=24a9e90d6458\", \"what_we_like\": \"Affordable, Snug fit\", \"best_for\": \"Budget-conscious parents\", \"price\": \"$18.00\", \"dimension\": \"\", \"weight\": \"\", \"image\": \"\", \"buy_link\": \"\"}, {\"name\": \"Halo Sleepsack 100% Cotton Muslin Wearable Blanket\", \"description\": \"The Halo Sleepsack 100% Cotton Muslin Wearable Blanket is considered the best luxury sleep sack. It is made of soft and breathable muslin fabric, providing a comfortable and safe sleeping environment for babies.\", \"source\": \"https://www.todaysparent.com/shopping/best-baby-sleep-sacks/\", \"what_we_like\": \"Soft and breathable muslin fabric, Luxurious feel\", \"best_for\": \"Parents looking for a high-end sleep sack\", \"price\": \"\", \"dimension\": \"\", \"weight\": \"\", \"image\": \"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQtDHIHMN7LLL7ismNOzyu7njoaRo9f4nsqqDz-eD_2HJVz_H5lqO9brGM&s\", \"buy_link\": \"\"}, {\"name\": \"Woolino 4 Season Ultimate Baby Sleep Bag\", \"description\": \"The Woolino 4 Season Ultimate Baby Sleep Bag is a versatile sleep sack suitable for all seasons. It is made of merino wool, which helps regulate body temperature and keeps babies comfortable throughout the night.\", \"source\": \"https://www.todaysparent.com/shopping/best-baby-sleep-sacks/\", \"what_we_like\": \"Suitable for all seasons, Made of merino wool\", \"best_for\": \"Parents looking for a versatile sleep sack\", \"price\": \"\", \"dimension\": \"\", \"weight\": \"\", \"image\": \"\", \"buy_link\": \"\"}, {\"name\": \"HALO Sleep Sack Baby Swaddle\", \"description\": \"The HALO Sleep Sack Baby Swaddle is considered the best overall sleep sack. It features a 3-way adjustable swaddle that helps babies feel secure and promotes better sleep.\", \"source\": \"https://www.parents.com/baby/care/newborn/best-sleep-sacks/\", \"what_we_like\": \"3-way adjustable swaddle, Promotes better sleep\", \"best_for\": \"Newborns and infants\", \"price\": \"$18.00\", \"dimension\": \"\", \"weight\": \"\", \"image\": \"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRj3soUkZU_QImq0ZsKsPhOlC5jOEF6cAxc8API1_iTMzfmVqEzpLg69gk&s\", \"buy_link\": \"\"}, {\"name\": \"Morrison Little Mo 20F Sleeping Bag\", \"description\": \"The Morrison Little Mo 20F Sleeping Bag is the best baby sleeping bag for camping. It is designed for cold weather and provides warmth and comfort during outdoor adventures.\", \"source\": \"https://momgoescamping.com/best-baby-sleeping-bags-camping/\", \"what_we_like\": \"Designed for cold weather, Suitable for camping or backpacking\", \"best_for\": \"Outdoor camping or backpacking\", \"price\": \"\", \"dimension\": \"\", \"weight\": \"\", \"image\": \"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRbDp0LFJVTrAiUFhBKfic7spa5tc56JDgzowNnuMlrIQcRVuV4X9KH3ns&s\", \"buy_link\": \"\"}, {\"name\": \"Halo SleepSack Wearable Blanket\", \"description\": \"The Halo SleepSack Wearable Blanket is the best baby sleep sack overall. It provides a safe and comfortable sleeping environment for babies and is available in various sizes and designs.\", \"source\": \"https://www.babycenter.com/baby-products/sleep/best-baby-sleep-sacks_40008018\", \"what_we_like\": \"Safe and comfortable, Available in various sizes and designs\", \"best_for\": \"All babies\", \"price\": \"$15.00\", \"dimension\": \"\", \"weight\": \"\", \"image\": \"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR7zWyYSj_qBK1DnugXbRoO7Qv-tzVJMGg7YEePNgeTNopfYH-rH1zGn_c&s\", \"buy_link\": \"\"}, {\"name\": \"Zippadee Zip Transitional Sleep Sack\", \"description\": \"The Zippadee Zip Transitional Sleep Sack is the best transitional sleep sack. It allows babies to move freely while still providing a secure and cozy sleeping environment.\", \"source\": \"https://www.whattoexpect.com/baby-products/sleep/best-sleep-sacks/\", \"what_we_like\": \"Allows freedom of movement, Secure and cozy\", \"best_for\": \"Babies transitioning from swaddling\", \"price\": \"$23.00\", \"dimension\": \"\", \"weight\": \"\", \"image\": \"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRwfNmWbb6nkaJitbQ0FYRroJE4nYTqqT81dJCWRCDWfuAFuzX5zBRltPc&s\", \"buy_link\": \"\"}, {\"name\": \"Primary Muslin Sleep Sack\", \"description\": \"The Primary Muslin Sleep Sack is the best runner-up sleep sack. It is made of soft and breathable muslin fabric, providing a comfortable and safe sleeping environment for babies.\", \"source\": \"https://www.verywellfamily.com/best-sleepsacks-4161094\", \"what_we_like\": \"Soft and breathable muslin fabric, Comfortable and safe\", \"best_for\": \"Parents looking for an alternative to the top pick\", \"price\": \"\", \"dimension\": \"\", \"weight\": \"\", \"image\": \"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSmDDNZUAkCkMCw7E2G84j8BzxLLuhuqIfqcE3FqZOLIcvtn5kBpDj6FBo&s\", \"buy_link\": \"\"}, {\"name\": \"Halo Cotton Muslin Sleepsack Wearable Blanket\", \"description\": \"The Halo Cotton Muslin Sleepsack Wearable Blanket is the best overall sleep sack. It is made of soft and breathable muslin fabric, providing a comfortable and safe sleeping environment for babies.\", \"source\": \"https://www.glamour.com/gallery/best-sleep-sacks\", \"what_we_like\": \"Soft and breathable muslin fabric, Comfortable and safe\", \"best_for\": \"All babies\", \"price\": \"$24.00\", \"dimension\": \"\", \"weight\": \"\", \"image\": \"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRk3-R09ilwxVdyvubWTVw0OT-KPJ-_rNkMLqGhISJbXnnMxAkqT-0QWd4&s\", \"buy_link\": \"\"}], \"sources\": [{\"title\": \"The Best Sleep Sacks To Keep Little Ones Cozy At Night - Forbes\", \"link\": \"https://www.forbes.com/sites/forbes-personal-shopper/article/best-sleep-sacks/?sh=24a9e90d6458\"}, {\"title\": \"Best Baby Sleep Sacks 2023 - Today's Parent\", \"link\": \"https://www.todaysparent.com/shopping/best-baby-sleep-sacks/\"}, {\"title\": \"The Best Sleep Sacks, Tested by Our Own Babies and Toddlers - Parents\", \"link\": \"https://www.parents.com/baby/care/newborn/best-sleep-sacks/\"}, {\"title\": \"The Best Baby Sleeping Bags for Camping\", \"link\": \"https://momgoescamping.com/best-baby-sleeping-bags-camping/\"}, {\"title\": \"Best baby sleep sacks of 2023 - BabyCenter\", \"link\": \"https://www.babycenter.com/baby-products/sleep/best-baby-sleep-sacks_40008018\"}, {\"title\": \"Best Sleep Sacks of 2023 — Best Wearable Blankets for Baby - What to Expect\", \"link\": \"https://www.whattoexpect.com/baby-products/sleep/best-sleep-sacks/\"}, {\"title\": \"The 13 Best Sleep Sacks of 2023, Tested and Reviewed - Verywell Family\", \"link\": \"https://www.verywellfamily.com/best-sleepsacks-4161094\"}, {\"title\": \"8 of the best baby sleeping bags 2023 - for newborns to 18 months | GoodTo\", \"link\": \"https://www.goodto.com/family/best-baby-sleeping-bags-631530\"}, {\"title\": \"10 Best Sleep Sacks for Newborns and Toddlers, According to Parents - Glamour\", \"link\": \"https://www.glamour.com/gallery/best-sleep-sacks\"}, {\"title\": \"The 8 Best Sleep Sacks of 2023, Tested and Reviewed by Parents - People\", \"link\": \"https://people.com/best-sleep-sacks-7106156\"}]}}\n{{\n  \"research_summary\": \"After conducting extensive research on the best baby sleeping bags, here are the top 10 products that are highly recommended:\",\n  \"items\": [\n    {\n      \"name\": \"Kyte Baby Sleep Bag\",\n      \"description\": \"The Kyte Baby Sleep Bag is considered the best overall sleep sack. It is made of high-quality materials and provides a cozy and comfortable sleeping environment for babies.\",\n      \"source\": \"https://www.forbes.com/sites/forbes-personal-shopper/article/best-sleep-sacks/?sh=24a9e90d6458\",\n      \"what_we_like\": \"Made of high-quality materials, Cozy and comfortable\",\n      \"best_for\": \"All babies\",\n      \"price\": \"$18.00\",\n      \"dimension\": \"\",\n      \"weight\": \"\",\n      \"image\": \"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRzLqCbeJXBtTvnbOyrJF29XPTkhaeGoi2r-At9jYiZLLLjo3jOc3kDxag&s\",\n      \"buy_link\": \"\"\n    },\n    {\n      \"name\": \"Burt's Bees Baby Beekeeper Blanket\",\n      \"description\": \"The Burt's Bees Baby Beekeeper Blanket is a value sleep sack that offers both comfort and affordability. It is made of 100% organic cotton and has a snug fit to keep babies secure during sleep.\",\n      \"source\": \"https://www.forbes.com/sites/forbes-personal-shopper/article/best-sleep-sacks/?sh=24a9e90d6458\",\n      \"what_we_like\": \"Affordable, Snug fit\",\n      \"best_for\": \"Budget-conscious parents\",\n      \"price\": \"$18.00\",\n      \"dimension\": \"\",\n      \"weight\": \"\",\n      \"image\": \"\",\n      \"buy_link\": \"\"\n    },\n    {\n      \"name\": \"Halo Sleepsack 100% Cotton Muslin Wearable Blanket\",\n      \"description\": \"The Halo Sleepsack 100% Cotton Muslin Wearable Blanket is considered the best luxury sleep sack. It is made of soft and breathable muslin fabric, providing a comfortable and safe sleeping environment for babies.\",\n      \"source\": \"https://www.todaysparent.com/shopping/best-baby-sleep-sacks/\",\n      \"what_we_like\": \"Soft and breathable muslin fabric, Luxurious feel\",\n      \"best_for\": \"Parents looking for a high-end sleep sack\",\n      \"price\": \"\",\n      \"dimension\": \"\",\n      \"weight\": \"\",\n      \"image\": \"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQtDHIHMN7LLL7ismNOzyu7njoaRo9f4nsqqDz-eD_2HJVz_H5lqO9brGM&s\",\n      \"buy_link\": \"\"\n    },\n    {\n      \"name\": \"Woolino 4 Season Ultimate Baby Sleep Bag\",\n      \"description\": \"The Woolino 4 Season Ultimate Baby Sleep Bag is a versatile sleep sack suitable for all seasons. It is made of merino wool, which helps regulate body temperature and keeps babies comfortable throughout the night.\",\n      \"source\": \"https://www.todaysparent.com/shopping/best-baby-sleep-sacks/\",\n      \"what_we_like\": \"Suitable for all seasons, Made of merino wool\",\n      \"best_for\": \"Parents looking for a versatile sleep sack\",\n      \"price\": \"\",\n      \"dimension\": \"\",\n      \"weight\": \"\",\n      \"image\": \"\",\n      \"buy_link\": \"\"\n    },\n    {\n      \"name\": \"HALO Sleep Sack Baby Swaddle\",\n      \"description\": \"The HALO Sleep Sack Baby Swaddle is considered the best overall sleep sack. It features a 3-way adjustable swaddle that helps babies feel secure and promotes better sleep.\",\n      \"source\": \"https://www.parents.com/baby/care/newborn/best-sleep-sacks/\",\n      \"what_we_like\": \"3-way adjustable swaddle, Promotes better sleep\",\n      \"best_for\": \"Newborns and infants\",\n      \"price\": \"$18.00\",\n      \"dimension\": \"\",\n      \"weight\": \"\",\n      \"image\": \"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRj3soUkZU_QImq0ZsKsPhOlC5jOEF6cAxc8API1_iTMzfmVqEzpLg69gk&s\",\n      \"buy_link\": \"\"\n    },\n    {\n      \"name\": \"Morrison Little Mo 20F Sleeping Bag\",\n      \"description\": \"The Morrison Little Mo 20F Sleeping Bag is the best baby sleeping bag for camping. It is designed for cold weather and provides warmth and comfort during outdoor adventures.\",\n      \"source\": \"https://momgoescamping.com/best-baby-sleeping-bags-camping/\",\n      \"what_we_like\": \"Designed for cold weather, Suitable for camping or backpacking\",\n      \"best_for\": \"Outdoor camping or backpacking\",\n      \"price\": \"\",\n      \"dimension\": \"\",\n      \"weight\": \"\",\n      \"image\": \"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRbDp0LFJVTrAiUFhBKfic7spa5tc56JDgzowNnuMlrIQcRVuV4X9KH3ns&s\",\n      \"buy_link\": \"\"\n    },\n    {\n      \"name\": \"Halo SleepSack Wearable Blanket\",\n      \"description\": \"The Halo SleepSack Wearable Blanket is the best baby sleep sack overall. It provides a safe and comfortable sleeping environment for babies and is available in various sizes and designs.\",\n      \"source\": \"https://www.babycenter.com/baby-products/sleep/best-baby-sleep-sacks_40008018\",\n      \"what_we_like\": \"Safe and comfortable, Available in various sizes and designs\",\n      \"best_for\": \"All babies\",\n      \"price\": \"$15.00\",\n      \"dimension\": \"\",\n      \"weight\": \"\",\n      \"image\": \"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR7zWyYSj_qBK1DnugXbRoO7Qv-tzVJMGg7YEePNgeTNopfYH-rH1zGn_c&s\",\n      \"buy_link\": \"\"\n    },\n    {\n      \"name\": \"Zippadee Zip Transitional Sleep Sack\",\n      \"description\": \"The Zippadee Zip Transitional Sleep Sack is the best transitional sleep sack. It allows babies to move freely while still providing a secure and cozy sleeping environment.\",\n      \"source\": \"https://www.whattoexpect.com/baby-products/sleep/best-sleep-sacks/\",\n      \"what_we_like\": \"Allows freedom of movement, Secure and cozy\",\n      \"best_for\": \"Babies transitioning from swaddling\",\n      \"price\": \"$23.00\",\n      \"dimension\": \"\",\n      \"weight\": \"\",\n      \"image\": \"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRwfNmWbb6nkaJitbQ0FYRroJE4nYTqqT81dJCWRCDWfuAFuzX5zBRltPc&s\",\n      \"buy_link\": \"\"\n    },\n    {\n      \"name\": \"Primary Muslin Sleep Sack\",\n      \"description\": \"The Primary Muslin Sleep Sack is the best runner-up sleep sack. It is made of soft and breathable muslin fabric, providing a comfortable and safe sleeping environment for babies.\",\n      \"source\": \"https://www.verywellfamily.com/best-sleepsacks-4161094\",\n      \"what_we_like\": \"Soft and breathable muslin fabric, Comfortable and safe\",\n      \"best_for\": \"Parents looking for an alternative to the top pick\",\n      \"price\": \"\",\n      \"dimension\": \"\",\n      \"weight\": \"\",\n      \"image\": \"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSmDDNZUAkCkMCw7E2G84j8BzxLLuhuqIfqcE3FqZOLIcvtn5kBpDj6FBo&s\",\n      \"buy_link\": \"\"\n    },\n    {\n      \"name\": \"Halo Cotton Muslin Sleepsack Wearable Blanket\",\n      \"description\": \"The Halo Cotton Muslin Sleepsack Wearable Blanket is the best overall sleep sack. It is made of soft and breathable muslin fabric, providing a comfortable and safe sleeping environment for babies.\",\n      \"source\": \"https://www.glamour.com/gallery/best-sleep-sacks\",\n      \"what_we_like\": \"Soft and breathable muslin fabric, Comfortable and safe\",\n      \"best_for\": \"All babies\",\n      \"price\": \"$24.00\",\n      \"dimension\": \"\",\n      \"weight\": \"\",\n      \"image\": \"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRk3-R09ilwxVdyvubWTVw0OT-KPJ-_rNkMLqGhISJbXnnMxAkqT-0QWd4&s\",\n      \"buy_link\": \"\"\n    }\n  ],\n  \"sources\": [\n    {\n      \"title\": \"The Best Sleep Sacks To Keep Little Ones Cozy At Night - Forbes\",\n      \"link\": \"https://www.forbes.com/sites/forbes-personal-shopper/article/best-sleep-sacks/?sh=24a9e90d6458\"\n    },\n    {\n      \"title\": \"Best Baby Sleep Sacks 2023 - Today's Parent\",\n      \"link\": \"https://www.todaysparent.com/shopping/best-baby-sleep-sacks/\"\n    },\n    {\n      \"title\": \"The Best Sleep Sacks, Tested by Our Own Babies and Toddlers - Parents\",\n      \"link\": \"https://www.parents.com/baby/care/newborn/best-sleep-sacks/\"\n    },\n    {\n      \"title\": \"The Best Baby Sleeping Bags for Camping\",\n      \"link\": \"https://momgoescamping.com/best-baby-sleeping-bags-camping/\"\n    },\n    {\n      \"title\": \"Best baby sleep sacks of 2023 - BabyCenter\",\n      \"link\": \"https://www.babycenter.com/baby-products/sleep/best-baby-sleep-sacks_40008018\"\n    },\n    {\n      \"title\": \"Best Sleep Sacks of 2023 — Best Wearable Blankets for Baby - What to Expect\",\n      \"link\": \"https://www.whattoexpect.com/baby-products/sleep/best-sleep-sacks/\"\n    },\n    {\n      \"title\": \"The 13 Best Sleep Sacks of 2023, Tested and Reviewed - Verywell Family\",\n      \"link\": \"https://www.verywellfamily.com/best-sleepsacks-4161094\"\n    },\n    {\n      \"title\": \"8 of the best baby sleeping bags 2023 - for newborns to 18 months | GoodTo\",\n      \"link\": \"https://www.goodto.com/family/best-baby-sleeping-bags-631530\"\n    },\n    {\n      \"title\": \"10 Best Sleep Sacks for Newborns and Toddlers, According to Parents - Glamour\",\n      \"link\": \"https://www.glamour.com/gallery/best-sleep-sacks\"\n    },\n    {\n      \"title\": \"The 8 Best Sleep Sacks of 2023, Tested and Reviewed by Parents - People\",\n      \"link\": \"https://people.com/best-sleep-sacks-7106156\"\n    }\n  ]\n}"