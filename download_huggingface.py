#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import AutoModel, AutoTokenizer
import os
import requests
from requests.exceptions import RequestException
import time

class HuggingFaceModel:
    def __init__(self, model_name="bert-base-uncased", cache_dir=None, max_retries=3):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.download_model()

    def download_model(self):
        retries = 0
        while retries < self.max_retries:
            try:
                # 设置缓存目录
                if self.cache_dir:
                    os.makedirs(self.cache_dir, exist_ok=True)
                
                # 使用AutoTokenizer下载并加载tokenizer
                # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir, trust_remote_code=True)
                # 使用AutoModel下载并加载模型
                self.model = AutoModel.from_pretrained(self.model_name, cache_dir=self.cache_dir, trust_remote_code=True)
                break  # 下载成功，退出循环
            except RequestException as e:
                print(f"Download error: {e}. Retrying...")
                retries += 1
                time.sleep(2)  # 等待2秒后重试
        if retries == self.max_retries:
            raise RuntimeError(f"Failed to download model after {self.max_retries} retries.")

    def preprocess_text(self, text):
        return self.tokenizer.encode_plus(text, return_tensors='pt')

    def get_model_prediction(self, input_data):
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model or tokenizer not loaded.")
        return self.model(**input_data)
    
if __name__ == "__main__":
    # 创建HuggingFaceModel实例，并指定缓存目录和最大重试次数
    hf_model = HuggingFaceModel("THUDM/chatglm-6b", cache_dir="/home/code/ChatGLM3_llma3/llama3_models", max_retries=5)
    
    # 确保指定的缓存目录存在
    if not os.path.exists("/home/code/ChatGLM3_llma3/llama3_models"):
        os.makedirs("/home/code/ChatGLM3_llma3/llama3_models")
    
    # 预处理文本
    # input_text = "Hello, world!"
    # preprocessed_data = hf_model.preprocess_text(input_text)
    
    # 获取模型预测
    # prediction = hf_model.get_model_prediction(preprocessed_data)
    
    # print(prediction)
        