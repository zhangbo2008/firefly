#= 从gpt2 debug学一下lm模型.
from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)  # 输出1,10, 768 输入 10 输出的部分前9个是之前句子的后9个字, 最后一个是新字. 
print(output)