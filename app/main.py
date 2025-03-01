from fastapi import FastAPI

app = FastAPI(title="Cloud Run API Boilerplate")

@app.get("/")
def read_root():
    return {"message": "Welcome to Cloud Run API Boilerplate!"}