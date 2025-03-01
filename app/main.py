from fastapi import FastAPI
from app.routes.process_10k import router as process_10k_router

app = FastAPI(title="Cloud Run API Boilerplate")

# Include routers
app.include_router(process_10k_router, prefix="/api/v1", tags=["10K Processing"])

@app.get("/")
def read_root():
    return {"message": "Welcome to Cloud Run API Boilerplate!"}