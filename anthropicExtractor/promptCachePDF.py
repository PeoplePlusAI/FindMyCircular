import base64
import json
import os
import re
import string

import anthropic
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from json_repair import repair_json
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
my_api_key = os.getenv("ANTHROPIC_API_KEY")

client = anthropic.Anthropic()

embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1.5",
    model_kwargs={"device": "cuda", "trust_remote_code": True},
)

vector_db = Chroma(
    collection_name="rag-chroma",
    persist_directory="/workspace/legalAgent/anthropicExtractor/chromaVDB",
    embedding_function=embeddings,
)


# PDF Processing
def pdf_processing(folder_path):
    pdf_files = os.listdir(folder_path)

    for pdf_file in pdf_files:
        with open(f"{folder_path}/{pdf_file}", "rb") as pdf_file:
            pdf_data = base64.standard_b64encode(pdf_file.read()).decode("utf-8")
            pdf_path = f"{pdf_file.name}"
            llm_processign(pdf_path, pdf_data)

    print("All PDFs processed.")


# LLM Calling
def llm_processign(pdf_path, pdf_data):
    message = client.beta.messages.create(
        model="claude-3-5-sonnet-20241022",
        betas=["pdfs-2024-09-25", "prompt-caching-2024-07-31"],
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": """
                        You are an advanced document analysis AI specializing in extracting structured information from official regulatory documents.

                        Task: Analyze the given document and generate a precise JSON structure with the following specifications:

                        Output Requirements:
                        - Use strict JSON formatting
                        - Ensure all extracted information is verbatim from the source document
                        - Maintain professional, neutral language
                        - Be comprehensive yet concise

                        Output Structure:
                        {
                            "name": "",
                            "date_of_issue": "",
                            "summary": "",
                            "relations": {
                                "document_name": "relationship_type"
                            },
                            "questions": []
                        }

                        1. Document Identification:
                        - Capture the exact, full official name of the document which is the unique name at the top of the document excluding the title
                        - Extract the precise date of issue

                        2. Document Relations Extraction:
                        - Identify all mentions of other documents within the text
                        - Extract the full official name of the related document, remove any additional information like dates for example "Circular 123" instead of "Circular 123 dated 2024"
                        - Determine their relationship of the extracted document to the current document using a single word. Examples: superseded, referenced, amended, invoked, replaced, cited

                        3. Optimized Summary:
                        - Create a concise yet comprehensive summary
                        - Ensure the summary is rich in key terms and contextual information
                        - Capture the core purpose, key regulatory changes, and fundamental implications of the document

                        4. Factual Questions:
                        - Generate questions that can be answered ONLY from the exact text of the document
                        - Questions must be:
                            * Directly answerable from the document's content
                            * Specific and precise
                            * Focused on factual details

                        
                    """,
                "cache_control": {"type": "ephemeral"},
            },
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_data,
                        },
                    },
                ],
            }
        ],
    )

    print(message.content[0].text)
    output_processing(pdf_path, message)


# JSON Processing and Saving
def output_processing(pdf_path, message):
    print(f"PDF Path: {pdf_path}")
    output_folder = "/workspace/legalAgent/anthropicExtractor/output"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    repaired_json = repair_json(message.content[0].text)
    output_data = json.loads(repaired_json)

    output_filename = f"{output_data['name']}.json"
    output_path = os.path.join(output_folder, output_filename)

    with open(output_path, "w") as json_file:
        json.dump(output_data, json_file, indent=4)

    print(f"Output saved to {output_path}")

    printable = set(string.printable)
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = result.document.export_to_markdown()
    doc = doc.replace("<!-- image -->", "")
    doc = "".join(filter(lambda x: x in printable, doc))
    doc = re.sub(r"\n+", "\n", doc)
    doc = re.sub(r"https?://\S+|www\.\S+", "", doc)
    doc = Document(
        page_content=doc,
        metadata={
            "source_path": pdf_path,
            "name": output_data["name"],
            "date_of_issue": output_data["date_of_issue"],
        },
    )
    ids = vector_db.add_documents(documents=[doc])
    print(f"Added document with ids: {ids}")


pdf_processing("/workspace/legalAgent/anthropicExtractor/circulars")
