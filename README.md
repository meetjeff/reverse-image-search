# Similar Image Search Service Documentation

This documentation explains how to operate a similar image search service developed using the vector database Milvus and the machine learning pipeline framework Towhee. The test image set is randomly selected from the ImageNet dataset, which includes a training set (`train`) and a test set (`test`). The training set contains 100 categories with 10 images each, and the test set contains 100 categories with 1 image each. There is a CSV file (`reverse_image_search.csv`) that includes basic information for 1,000 images from the training set, such as the image ID, path, and category.

## 1. Create Vector Database
I used Zilliz Cloud, the cloud-hosted version of Milvus:
- Register and obtain `CLUSTER_ENDPOINT` and `TOKEN`.
- Set these as the values for the environment variables `MILVUS_URI` and `MILVUS_TOKEN`.
  - [Register with Zilliz Cloud](https://docs.zilliz.com/docs/register-with-zilliz-cloud)

## 2. Install Dependencies (via pipenv)
```bash
pip3 install pipenv
pipenv install
```

## 3. Create Milvus Collection
Invoke the create_collection() method:
```bash
pipenv run python prepare/create_collection.py
# Or, if setting PYTHONPATH
$env:PYTHONPATH="src"; python prepare/create_collection.py
```
* Set the created collection_name as the value for the environment variable IMAGE_COLLECTION.
* Optionally, use the drop_collection() method to delete a Collection if needed.

## 4. Select the Model for Image Vectorization
For example, use resnet50 and set it as the value for the environment variable IMAGE_MODEL.

## 5. Vectorize Images in images/train Directory Listed in reverse_image_search.csv and Load into Milvus Collection
```bash
pipenv run python prepare/load_data.py
# Or, if setting PYTHONPATH
$env:PYTHONPATH="src"; python prepare/load_data.py
```

## 6. Start the Application (Uvicorn server)
```bash
pipenv run uvicorn main:app --reload
# Or, if setting PYTHONPATH
$env:PYTHONPATH="src"; uvicorn main:app --reload
```  