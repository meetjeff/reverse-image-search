import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from milvus_manager import MilvusManager, set_milvus_client
from embedding import EmbeddingService

MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
IMAGE_COLLECTION = os.getenv("IMAGE_COLLECTION")
IMAGE_MODEL = os.getenv("IMAGE_MODEL")
DEVICE = os.getenv("DEVICE")

milvus_client = set_milvus_client(MILVUS_URI, MILVUS_TOKEN)
milvus_manager = MilvusManager(milvus_client)
embedding_service = EmbeddingService(milvus_manager, IMAGE_COLLECTION, IMAGE_MODEL, DEVICE)

app = FastAPI()

class ImageData(BaseModel):
    image: str
    id: int
    label: str = None

@app.get('/')
def home():
    return {'test':'test'}

@app.get('/search-image')
async def search_images(image: str, limit: int=3):
    try:
        return embedding_service.image_search(image, limit)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post('/insert-image')
async def insert_images(data: ImageData):
    try:
        embedding_service.upsert_milvus(data.id, data.image, data.label)
        return {'status': True, 'msg': 'Success'}, 200
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
