from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow requests from React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Allow React frontend to make requests
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/items")
def get_items():
    return [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}, {"id": 3, "name": "Item 3"}]

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)




 # for job in jobs:  # ✅ CORRECT! Now looping over the actual job postings
        #     if not isinstance(job, dict):  
        #         logging.warning(f"Invalid job format: {job}")
        #         continue

        #     skills = job.get("skills", [])
        #     if not isinstance(skills, list):
        #         logging.warning(f"Skills is not a list for job: {job}")
        #         skills = []  # Default to empty list

        #     links = portfolio.query_links(skills)
        #     email = chain.write_email(job, links)

        #     results.append({"job": job, "email": email})

        # return {"results": results}