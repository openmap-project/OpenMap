import numpy as np
import open3d as o3d
import openai
from omegaconf import OmegaConf, DictConfig
from utils.time_utils import timeit

class Dialogue:
    def __init__(self, cfg: DictConfig):
        self.client = openai.OpenAI(
            api_key=cfg.OPENAI_KEY,
            base_url=cfg.base_url,
        )
        self.messages = []

    def sys_say(self, text):
        self.messages.append({"role": "system", "content": text})
    
    @timeit
    def user_query(self, text):
        self.messages.append({"role": "user", "content": text})
        response = self.client.chat.completions.create(
            model="deepseek-v3",
            messages=self.messages,
        )
        resp = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": resp})
        return resp

    def clear(self,):
        self.messages = []

    def save(self, save_path):
        with open(save_path, "a+") as f:
            for message in self.messages:
                f.write(f"{message['role']}: {message['content']}\n")
