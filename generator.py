import torch
import time
import json
import os
from tqdm import tqdm
from openai import OpenAI
from pymilvus import DataType, MilvusClient
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

class Generator:
    def __init__(self, config):
        self.embed_model = config.embedding.embed_model
        self.dim = config.embedding.dim
        self.uri = f"http://{config.milvus.host}:{config.milvus.port}"
        self.collection_name = config.milvus.collection_name
        self.limit = config.milvus.limit
        # 从环境变量中读取deepseek的api key
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        # 创建Milvus实例
        self.milvus_client = MilvusClient(uri=self.uri)
        # 创建openai客户端实例
        self.OpenAI_client = OpenAI(api_key=self.deepseek_api_key, base_url="https://api.deepseek.com")
        self.model = config.llm.model
        self.temperature = config.llm.temperature
 

    def create_collection(self, collection_name):
        """创建集合"""
        # 检查同名集合是否存在，如果存在则删除，不存在则创建
        if self.milvus_client.has_collection(self.collection_name):
            print(f"集合 {self.collection_name} 已经存在")
            try:
                self.milvus_client.drop_collection(self.collection_name)
                print(f"删除集合：{self.collection_name}")
            except Exception as e:
                print(f"删除集合时出现错误: {e}")
        # 创建集合模式
        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
            num_partitions=16,
            description=""
        )
        # 添加字段到schema
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=256)
        schema.add_field(field_name="book", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="author", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="type", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="date", datatype=DataType.VARCHAR, max_length=32)
        schema.add_field(field_name="chunk", datatype=DataType.VARCHAR, max_length=2048)
        schema.add_field(field_name="window", datatype=DataType.VARCHAR, default_value="", max_length=6144)
        schema.add_field(field_name="method", datatype=DataType.VARCHAR, max_length=32)
        schema.add_field(field_name="dense_vectors", datatype=DataType.FLOAT_VECTOR, dim=512)
        # 创建集合
        try:
            self.milvus_client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                shards_num=2
            )
            print(f"创建集合：{self.collection_name}")
        except Exception as e:
            print(f"创建集合的过程中出现了错误: {e}")
        # 等待集合创建成功
        while not self.milvus_client.has_collection(self.collection_name):
            time.sleep(1)
        print(f"集合 {self.collection_name} 创建成功")

    def vectorize_query(self, query):
        """向量化文本列表"""
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        use_fp16 = device.startswith("cuda")
        bge_m3_ef = BGEM3EmbeddingFunction(
            model_name=self.embed_model,
            device=device,
            use_fp16=use_fp16
        )
        vectors = bge_m3_ef.encode_documents(query)
        return vectors

    def get_files_from_dir(self, input_dir_path):
        file_paths = []
        # 遍历目录中的所有文件和子目录
        for root, dirs, files in os.walk(input_dir_path):
            for file in files:
                # 检查文件扩展名是否为 .json
                if file.endswith('.json'):
                    # 构建文件的完整路径
                    file_path = os.path.join(root, file)
                    file_paths.append(file_path)
        return file_paths

    def vectorize_and_import_data(
        self, 
        input_file_path, 
        field_name, 
        embed_model,
        batch_size
        ):
        """读取json文件中的数据，向量化后，分批入库"""
        with open(input_file_path, 'r', encoding='utf-8') as file:
            data_list = json.load(file)
            # data_list = data_list[:1000]
            # 提取该json文件中的所有指定字段的值
            query = [data[field_name] for data in data_list]
        # 向量化文本数据
        vectors = self.vectorize_query(query)
        for data, dense_vectors in zip(data_list, vectors['dense']):
            data['dense_vectors'] = dense_vectors.tolist()
        print(f"正在将数据插入集合：{self.collection_name}")
        total_count = len(data_list)
        with tqdm(total=total_count, desc="插入数据") as pbar:
            for i in range(0, total_count, batch_size):
                batch_data = data_list[i:i + batch_size]
                res = self.milvus_client.insert(
                    collection_name=self.collection_name,
                    data=batch_data
                )
                pbar.update(len(batch_data))

    def create_index(self, collection_name):
        """创建索引"""
        # 创建索引参数
        index_params = self.milvus_client.prepare_index_params()
        index_params.add_index(
            index_name="IVF_FLAT",
            field_name="dense_vectors",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128}
        )
        # 创建索引
        self.milvus_client.create_index(
            collection_name=self.collection_name,
            index_params=index_params
        )
        print("索引创建完成")
        # 加载集合
        print(f"正在加载集合：{self.collection_name}")
        self.milvus_client.load_collection(collection_name=self.collection_name)
        state = str(self.milvus_client.get_load_state(collection_name=self.collection_name)['state'])
        # 验证加载状态
        if state == 'Loaded':
            print("集合加载完成")
        else:
            print("集合加载失败")

    def create_vector_db(self):
        """创建向量数据库"""
        # 创建集合
        self.create_collection(self.collection_name)
        # 入库
        start_time = time.time()
        batch_size = 1000
        field_name = "chunk"
        input_dir_path = 'data'
        input_file_paths = self.get_files_from_dir(input_dir_path)
        for input_file_path in input_file_paths:
            print(f"正在处理文件：{input_file_path}")
            self.vectorize_and_import_data(input_file_path, field_name, self.embed_model, batch_size)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"数据入库耗时：{elapsed_time:.2f} 秒")
        # 创建索引
        self.create_index(self.collection_name)

    def search(self, query):
        """搜索"""
        # 创建搜索参数
        search_params = {
            "metric_type": "COSINE",
            "params": {
                "nprobe": 16,
                "radius": 0.1,
                "range_filter": 1
            }
        }
        # 搜索向量
        query_vectors = [self.vectorize_query([query])['dense'][0].tolist()]
        res = self.milvus_client.search(
            collection_name=self.collection_name,
            data=query_vectors,
            anns_field="dense_vectors",
            search_params=search_params,
            limit=self.limit,
            output_fields=["window", "title"]
        )
        return res

    def get_ref_info(self, query):
        """从搜索结果中提取参考信息"""
        ref_info_list = []
        res = self.search(query)
        for hit in res[0]:
            ref_info = {
                "ref": hit["entity"]["window"],
                "title": hit["entity"]["title"]
            }
            ref_info_list.append(ref_info)
        return ref_info_list
    
    def generate_response(self, query):
        """成响应函数"""
        ref_info_list = self.get_ref_info(query)
        system_prompt = "你是鲁迅作品研究者，熟悉鲁迅的各种作品。"
        user_prompt = (
            f"请你根据提供的参考信息，查找是否有与问题语义相似的内容。参考信息：{ref_info_list}。问题：{query}。\n"
            f"如果找到了相似的内容，请回复“鲁迅的确说过类似的话，原文是[原文内容]，这句话来自[文章标题]”。\n"
            f"[原文内容]是参考信息中ref字段的值，[文章标题]是参考信息中title字段的值。如果引用它们，请引用完整的内容。\n"
            f"如果参考信息没有提供和问题相关的内容，请回答“据我所知，鲁迅并没有说过类似的话。”"
        )
        response = self.OpenAI_client.chat.completions.create(
            model = self.model,
            messages=[
                # 设置系统信息，通常用于设置模型的行为、角色或上下文。
                {"role": "system", "content": system_prompt},
                # 设置用户消息，用户消息是用户发送给模型的消息。
                {"role": "user", "content": user_prompt},
            ],
            # 设置温度参数，数值在0-2之间，默认值为 1.0。值越高，生成的文本越随机；值越低，生成的文本越确定
            temperature = self.temperature,  
            stream = True
        )
        # 遍历响应中的每个块
        for chunk in response:
            # 检查块中是否包含选择项
            if chunk.choices:
                # 打印选择项中的第一个选项的增量内容，并确保立即刷新输出
                print(chunk.choices[0].delta.content, end="", flush=True)
        print("\n")
        # print("*"*100)
        # print(ref_info_list)

    def delete_collection(self):
        if self.milvus_client.has_collection(self.collection_name):
            try:
                self.milvus_client.drop_collection(self.collection_name)
                print(f"删除集合：{self.collection_name}")
            except Exception as e:
                print(f"删除集合时出现错误: {e}")
        else:
            print(f"集合 {self.collection_name} 不存在，无需删除。")