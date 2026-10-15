from fastapi import FastAPI

# Creating FastAPI object
app = FastAPI()

@app.get("/health")
def get_health():
    return {"status": "ok"}


# Load model
