import base64
import json
import os
from datetime import datetime

import anthropic
from dotenv import load_dotenv

load_dotenv()
my_api_key = os.getenv("ANTHROPIC_API_KEY")

client = anthropic.Anthropic()

with open(
    "/workspace/legalAgent/anthropicExtractor/circulars/Existing Circular.pdf", "rb"
) as pdf_file:
    pdf_base64 = base64.standard_b64encode(pdf_file.read()).decode("utf-8")

response = client.beta.messages.count_tokens(
    betas=["token-counting-2024-11-01", "pdfs-2024-09-25"],
    model="claude-3-5-sonnet-20241022",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf_base64,
                    },
                },
                {"type": "text", "text": "Please summarize this document."},
            ],
        }
    ],
)

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
output_filename = f"response_{timestamp}.json"

with open(output_filename, "w") as json_file:
    json.dump(response.json(), json_file, indent=4)

print(response.json())
