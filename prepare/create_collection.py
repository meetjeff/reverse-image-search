import os
from pymilvus import MilvusClient, DataType

def create_collection():
    MILVUS_URI = os.getenv("MILVUS_URI")
    MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")

    client = MilvusClient(
        uri=MILVUS_URI,
        token=MILVUS_TOKEN 
    )

    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
    )
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="image_vector", datatype=DataType.FLOAT_VECTOR, dim=2048)
    schema.add_field(field_name="path", datatype=DataType.VARCHAR, max_length=1000)
    schema.add_field(field_name="label", datatype=DataType.VARCHAR, max_length=50)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="id"
    )
    index_params.add_index(
        field_name="image_vector", 
        index_type="AUTOINDEX",
        metric_type="L2"
    )

    client.create_collection(
        collection_name="post_image",
        schema=schema,
        index_params=index_params
    )

    print('collection has been created.')


if __name__ == "__main__":
    create_collection()