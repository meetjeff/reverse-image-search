import os

from milvus_manager import MilvusManager, set_milvus_client

def drop_collection():
    MILVUS_URI = os.getenv("MILVUS_URI")
    MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
    COLLECTION_TO_DROP = os.getenv("COLLECTION_TO_DROP", "post_image")

    milvus_client = set_milvus_client(MILVUS_URI, MILVUS_TOKEN)
    milvus_manager = MilvusManager(milvus_client)
    milvus_manager.drop_collection(COLLECTION_TO_DROP)

    print(f'{COLLECTION_TO_DROP} has been dropped.')


if __name__ == "__main__":
    drop_collection()