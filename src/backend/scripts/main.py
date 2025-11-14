from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from controller import post_imgs

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return "Select img to start"

# Route từ controller

app.post("/api/post-imgs")(post_imgs)
