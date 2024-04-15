from pymilvus import MilvusClient

def set_milvus_client(milvus_cluster_endpoint, milvus_token):
    return MilvusClient(
        uri=milvus_cluster_endpoint,
        token=milvus_token 
    )

class MilvusManager:
    def __init__(self, milvus_client):
        self.client = milvus_client

    def drop_collection(self, collection_name: str):
        self.client.drop_collection(
            collection_name=collection_name
        )

    def count_collection_data(self, collection_name: str):
        return self.client.query(
            collection_name=collection_name,
            output_fields=["count(*)"]
        )[0]["count(*)"]

    def upsert_data(self, collection_name: str, data: list[dict]):
        self.client.upsert(
            collection_name=collection_name,
            data=data
        )

    def vectors_search(
            self, 
            collection_name: str, 
            query_vectors: list, 
            limit: int, 
            output_fields: list=None
        ) -> list:
        return self.client.search(
            collection_name=collection_name,
            data=query_vectors,
            limit=limit,
            output_fields = output_fields
        )
    
    def get_data_by_ids(self, collection_name: str, ids: list):
        return self.client.get(
            collection_name=collection_name,
            ids=ids
        )
    
    def query_data(self, collection_name:str, query_filter:str, output_fields:list, limit:int):
        return self.client.query(
            collection_name=collection_name,
            filter=query_filter,
            output_fields=output_fields,
            limit=limit
        )