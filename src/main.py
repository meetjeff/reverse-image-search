import os
from fastapi import FastAPI, HTTPException, status, Query
from fastapi.responses import RedirectResponse
from typing import List

from milvus_manager import MilvusManager, set_milvus_client
from embedding import EmbeddingService
from params_model import ImageData, QueryParams

MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
IMAGE_COLLECTION = os.getenv("IMAGE_COLLECTION")
IMAGE_MODEL = os.getenv("IMAGE_MODEL")
DEVICE = os.getenv("DEVICE")

milvus_client = set_milvus_client(MILVUS_URI, MILVUS_TOKEN)
milvus_manager = MilvusManager(milvus_client)
embedding_service = EmbeddingService(
    milvus_manager, 
    IMAGE_COLLECTION, 
    IMAGE_MODEL, 
    DEVICE
)

app = FastAPI()

@app.get("/")
async def root():
    return RedirectResponse(
        url='/docs', 
        status_code=status.HTTP_307_TEMPORARY_REDIRECT
    )

@app.get('/search-image')
async def search_image(image: str, limit: int=3):
    try:
        return embedding_service.image_search(image, limit)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post('/insert-image')
async def insert_image(image_data: ImageData):
    try:
        embedding_service.upsert_milvus(
            image_data.id, 
            image_data.image, 
            image_data.label
        )
        return {'status': True, 'msg': 'Success'}, 200
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.get('/count-collection')
async def count_collection(collection: str):
    try:
        return milvus_manager.count_collection_data(collection)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.get('/get-data-by-ids')
async def get_data_by_ids(
    collection: str, 
    id: List[int] = Query(...), 
    output_field: List[str] = Query(default=['id'])
):
    try:
        return milvus_manager.get_data_by_ids(collection, id, output_field)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post('/query-data')
async def query_data(query_params: QueryParams):
    try:
        return milvus_manager.query_data(
            query_params.collection, 
            query_params.query_filter, 
            query_params.output_fields, 
            query_params.limit
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))