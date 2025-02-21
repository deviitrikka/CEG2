from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_community.document_loaders import WebBaseLoader
import os
import chromadb
import pandas as pd
import uuid
import re
import logging
import json

logging.basicConfig(level=logging.INFO)
# Load environment variables
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("Missing GROQ_API_KEY in environment variables")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class URLInput(BaseModel):
    url: str

class Chain:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile"
        )

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
        """
        ### SCRAPED TEXT FROM WEBSITE
        {page_data}
        
        ### INSTRUCTION:
        Extract job postings from the scraped text and return a JSON object with:
        - `role`
        - `skills` (as a list)
        - `experience`
        - `job_details`

        **Ensure that the response is valid JSON and does not contain extra text.**

        ### OUTPUT FORMAT:
        ```json
        {{
            "role": "Software Engineer",
            "skills": ["Python", "Django", "REST APIs"],
            "experience": "2+ years",
            "job_details": "Full-time position in a tech company."
        }}
        ```
        """
    )

        chain = prompt_extract | self.llm
        response = chain.invoke(input={'page_data': cleaned_text})
        print(response.content) #added
        try:
            return JsonOutputParser().parse(response.content)
        except OutputParserException:
            raise HTTPException(status_code=400, detail="Context too big. Unable to parse jobs.")

    
    def write_email_as_ind(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are ABC, a highly skilled professional with expertise relevant to the given job description. 
            You are applying for this role and need to craft a compelling, personalized cold email that highlights:
            - Your key skills and experience that align with the job.
            - Your enthusiasm for the position and the company.
            - How you can add value to the organization.
            - A polite and professional request for further discussion or an interview.

            Make the email concise, engaging, and persuasive while maintaining a formal yet approachable tone.
            Use the relevant links provided to support your qualifications.

            ### OUTPUT FORMAT:
            Provide the email in a professional format with:
            - A subject line
            - A proper salutation
            - A well-structured body (opening, main content, closing)
            - A polite sign-off

            ### EMAIL (NO PREAMBLE):
            """
        )

        chain_email = prompt_email | self.llm
        print(chain_email.invoke({"job_description": str(job), "link_list": links}).content)
        return chain_email.invoke({"job_description": str(job), "link_list": links}).content
    
    def write_email_as_be(self, job, links):
            prompt_email = PromptTemplate.from_template(
                """
                ### JOB DESCRIPTION:
                {job_description}

                ### INSTRUCTION:
                You are **[Business Development Executive]** at ABCD Company, reaching out to company provided in job description regarding their IT staffing needs. 
                Write a **professional cold email** offering Infosys' IT staff to fulfill the mentioned company' vacancies, 
                emphasizing **cost savings, expertise, and flexibility**.


                ### OUTPUT FORMAT:
                Provide the email in a professional format with:
                - A subject line
                - A proper salutation
                - A well-structured body (opening, main content, closing) with indiviuals you have including their profile links related to the skills in job description from {link_list}
                - A polite sign-off

                ### EMAIL (NO PREAMBLE):
                """
            )

            chain_email = prompt_email | self.llm
            print(chain_email.invoke({"job_description": str(job), "link_list": links}).content)
            return chain_email.invoke({"job_description": str(job), "link_list": links}).content

class Portfolio:
    def __init__(self, file_name="my_portfolio.csv"):
        self.file_path = os.path.join("resources", file_name)
        if os.path.exists(self.file_path):
            self.df = pd.read_csv(self.file_path)
        else:
            self.df = pd.DataFrame(columns=["Techstack", "Links"])
        self.client = chromadb.PersistentClient('vectorstore')
        self.collection = self.client.get_or_create_collection(name="portfolio")

    def load_portfolio(self):
        if not self.collection.count():
            for _, row in self.df.iterrows():
                print(row["Techstack"], row["Links"])
                self.collection.add(documents=row["Techstack"],
                                    metadatas={"links": row["Links"]},
                                    ids=[str(uuid.uuid4())])
    
    def query_links(self, skills):
        result = self.collection.query(query_texts=skills, n_results=2).get('metadatas', [])
        print(result)
        return result

def clean_text(text):
    text = re.sub(r'<[^>]*?>', '', text)
    text = re.sub(r'http[s]?://[^\s]+', '', text)
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text

chain = Chain()
portfolio = Portfolio()

@app.post("/process1")
def process_url(data: URLInput):
    try:
        logging.info(f"Processing URL: {data.url}")
        
        # Scrape and clean the text
        loader = WebBaseLoader([data.url])
        text_data = clean_text(loader.load().pop().page_content)
        logging.info(f"Cleaned text: {text_data[:100]}...")  # Log only the first 100 chars
        
        # Load portfolio
        portfolio.load_portfolio()
        logging.info("Portfolio loaded successfully.")
        
        # Extract jobs
        job = chain.extract_jobs(text_data) #shd return a dict
        # logging.info(f"Extracted jobs: {jobs}")
        # print(job)
        # job_list = jobs["job_postings"] #added2
        # try:
        #     job = json.loads(job)  # Ensure it's a dictionary
        # except json.JSONDecodeError:
        #     raise HTTPException(status_code=400, detail="Failed to parse job data")
        # Process jobs
        results = []

       
        skills = job.get("skills", []) #
        if not isinstance(skills, list):
            logging.warning(f"Skills is not a list for job: {job}")
            skills = []  # Default to empty list
        
        links = portfolio.query_links(skills)
        if not isinstance(links, list):
            raise HTTPException(status_code=500, detail="Portfolio query did not return a list")

        # email = chain.write_email(job, links)
        email = chain.write_email_as_ind(json.dumps(job), links)


        results.append({"job": job, "email": email})
        return {"results": results}
    
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/process2")
def process_url(data: URLInput):
    try:
        logging.info(f"Processing URL: {data.url}")
        
        # Scrape and clean the text
        loader = WebBaseLoader([data.url])
        text_data = clean_text(loader.load().pop().page_content)
        logging.info(f"Cleaned text: {text_data[:100]}...")  # Log only the first 100 chars
        
        # Load portfolio
        portfolio.load_portfolio()
        logging.info("Portfolio loaded successfully.")
        
        # Extract jobs
        job = chain.extract_jobs(text_data) #shd return a dict
        # logging.info(f"Extracted jobs: {jobs}")
        # print(job)
        # job_list = jobs["job_postings"] #added2
        # try:
        #     job = json.loads(job)  # Ensure it's a dictionary
        # except json.JSONDecodeError:
        #     raise HTTPException(status_code=400, detail="Failed to parse job data")
        # Process jobs
        results = []

       
        skills = job.get("skills", []) #
        if not isinstance(skills, list):
            logging.warning(f"Skills is not a list for job: {job}")
            skills = []  # Default to empty list
        
        links = portfolio.query_links(skills)
        if not isinstance(links, list):
            raise HTTPException(status_code=500, detail="Portfolio query did not return a list")

        # email = chain.write_email(job, links)
        email = chain.write_email_as_be(json.dumps(job), links)


        results.append({"job": job, "email": email})
        return {"results": results}
    
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
