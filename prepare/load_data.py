import os
import pandas as pd

from milvus_manager import MilvusManager, set_milvus_client
from embedding import EmbeddingService

def convert_path(path:str, root_folder: str='.'):
    return root_folder + path[2:]

def batch_upsert_milvus(
        image_list, 
        root_folder, 
        embedding_service, 
        milvus_manager, 
        collection, 
        batch_size=100
    ):
    df = pd.read_csv(root_folder + image_list)
    for start in range(0, len(df), batch_size):
        end = start + batch_size
        batch_df = df[start:end]
        data = [{
            'id': img['id'], 
            'image_vector': embedding_service.get_image_vector(
                convert_path(img['path'], root_folder)
            ),
            'path': img['path'], 
            'label': img['label']
        } for _, img in batch_df.iterrows()]
        milvus_manager.upsert_data(collection, data)

def load_data():
    MILVUS_URI = os.getenv("MILVUS_URI")
    MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
    IMAGE_COLLECTION = os.getenv("IMAGE_COLLECTION")
    IMAGE_MODEL = os.getenv("IMAGE_MODEL")
    DEVICE = os.getenv("DEVICE")
    IMAGE_FOLDER = os.getenv("IMAGE_FOLDER", "images/")
    IMAGE_LIST_CSV = os.getenv("IMAGE_LIST_CSV", "reverse_image_search.csv")

    milvus_client = set_milvus_client(MILVUS_URI, MILVUS_TOKEN)
    milvus_manager = MilvusManager(milvus_client)
    embedding_service = EmbeddingService(
        milvus_manager, 
        IMAGE_COLLECTION, 
        IMAGE_MODEL, 
        DEVICE
    )
    batch_upsert_milvus(
        IMAGE_LIST_CSV, 
        IMAGE_FOLDER, 
        embedding_service, 
        milvus_manager, 
        IMAGE_COLLECTION
    )

    print(f'{IMAGE_LIST_CSV} has been upserted to milvus')


if __name__ == "__main__":
    load_data()