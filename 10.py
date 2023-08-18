#===========看看数据集的构建. firefly里面的数据集如何构建,和数据的格式问题

from transformers import AutoTokenizer, BitsAndBytesConfig
from component.collator import SFTDataCollator
from component.dataset import SFTDataset, ChatGLM2SFTDataset
tokenizer = AutoTokenizer.from_pretrained(
    "THUDM/chatglm2-6b",
    trust_remote_code=True,
    # llama不支持fast
    use_fast= True
)


a=ChatGLM2SFTDataset("./data/dummy_data.jsonl", tokenizer, 1024)
b=a.__getitem__(0)