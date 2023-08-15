from transformers import pipeline, set_seed # 学gpt2模型.
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
generator("The White man worked as a", max_length=10, num_return_sequences=5)
# modeling_gpt2.py:1100 行可以看出来lm模型的含义和loss计算. 每一次喂入 0,1,2,3,4 输出 1^,2^,3^,4^,5^  然后我们把 这5个跟(1,2,3,4,5)做交叉熵即可.

set_seed(42)
generator("The Black man worked as a", max_length=10, num_return_sequences=5)
