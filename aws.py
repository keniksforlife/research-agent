import json
import boto3
from botocore.signers import RequestSigner
import requests

# AWS credentials
aws_access_key = 'AKIAI2OGJGO5PPPX44PA'
aws_secret_access_key = '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'
aws_session_token = ''  # Optional

# Service and region
service = 'ProductAdvertisingAPI'
region = 'us-east-1'

# Initialize request signer
session = boto3.Session(
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_access_key,
    aws_session_token=aws_session_token,
    region_name=region
)
signer = RequestSigner(service, session)

# API endpoint and request parameters
host = 'webservices.amazon.com'
endpoint = f'https://{host}/paapi5/searchitems'
method = 'POST'
headers = {
    'Content-Type': 'application/json',
    'X-Amz-Target': 'com.amazon.paapi5.v1.ProductAdvertisingAPIv1.SearchItems',
    'User-Agent': 'python-requests/1.0.0'
}
payload = {
    'Marketplace': 'www.amazon.com',
    'PartnerType': 'Associates',
    'PartnerTag': 'onamzwwwfanfi-21',
    'Keywords': 'kindle',
    'SearchIndex': 'All',
    'ItemCount': 3,
    'Resources': ['Images.Primary.Large', 'ItemInfo.Title', 'Offers.Listings.Price']
}
body = json.dumps(payload)

# Sign the request
params = {'method': method, 'url': endpoint, 'headers': headers, 'data': body}
signed_headers = signer.generate_presigned_url(
    method,
    endpoint,
    region,
    service,
    request_dict=params
)

# Make the API request
response = requests.post(endpoint, headers=signed_headers, data=body)

# Output the results
if response.status_code == 200:
    print("Success:", json.dumps(response.json(), indent=2))
else:
    print("Error:", response.content)
