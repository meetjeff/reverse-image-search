from towhee import pipe, ops

class EmbeddingService:
    def __init__(self, vector_db, collection, image_model, device):
        self.vector_db = vector_db
        self.collection = collection
        self.image_model = image_model
        self.device = device
        self.image_vector_pipe = self.__get_image_vector_pipe()
        self.detect_embedding_pipe = self.__detect_embedding_pipe()

    @staticmethod
    def __image_decode_pipe():
        return pipe.input('image_path').map(
            'image_path', 
            'image_array', 
            ops.image_decode()
        )
    
    def __image_embedding_pipe(self):
        return (self.__image_decode_pipe().map(
            'image_array', 
            'image_vector', 
            ops.image_embedding.timm(model_name=self.image_model, device=self.device)
        ))

    def __get_image_vector_pipe(self):
        return self.__image_embedding_pipe().output('image_vector')

    def __detect_embedding_pipe(self): 
        return (
            self.__image_embedding_pipe()
            .map('image_array', ('box', 'class', 'score'), ops.object_detection.yolov5())
            .map(('image_array', 'box'), 'object', ops.image_crop(clamp=True))
            .map('object', 'object_vector', ops.image_embedding.timm(model_name=self.image_model, device=self.device))
            .output('class', 'object_vector', 'image_vector')
        )
    
    @staticmethod
    def __flatten_dict(data, keys):
        new_dict = {'id': data['id']}
        entity = data.get('entity', {})
        for key in keys:
            if key in entity:
                new_dict[key] = entity[key]
        
        return new_dict

    def upsert_milvus(self, id, image_path, label='None'):
        data = [{
            'id': id, 
            'image_vector': self.image_vector_pipe(image_path).get()[0],
            'path': image_path, 
            'label': label
        }]
        self.vector_db.upsert_data(self.collection, data)

    def search_milvus(self, query_vectors, limit, output_fields):
        res = self.vector_db.vectors_search(self.collection, query_vectors, limit, output_fields)
        return [[self.__flatten_dict(d, output_fields) for d in l] for l in res]

    def image_search(self, image_path, limit):
        detect_res = self.detect_embedding_pipe(image_path).get()
        objects = set(detect_res[0])
        query_vectors = (detect_res[1] if isinstance(detect_res[1], list) else [detect_res[1]]) + [detect_res[2]]
        output_fields = ['label']
        search_res = self.search_milvus(query_vectors, limit, output_fields)
        images = {image['id']: image['label'] for sublist in search_res for image in sublist}
        return {
            'objects': objects,
            'images': images
        }