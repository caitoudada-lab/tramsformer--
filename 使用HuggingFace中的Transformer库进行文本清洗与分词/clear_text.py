import re
import string
from transformers import BertTokenizer, BertModel
import torch

def clean_text(text):
    # 转换为小写
    text = text.lower()
    # re.sub(pattern, repl, string) pattern：该参数表示正则中的模式字符串
    # repl 要替换的字符串  string 要处理的文本
    # re.escape() 能够对字符串中的所有的特殊字符进行转义。因为正则表达式中. * +有特殊的含义
    # 要想当作普通符号进行处理，需要转义。
    # string.punctuation  string模块中一个常量，它含有了所有ASCII字符中的标点符号。
    # Python 中 它的值通常是 !"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub('\s+', ' ', text).strip()
    return text

# 加载分词器和模型
tokenizer = BertTokenizer.from_pretrained('./bert_localpath/')
bert = BertModel.from_pretrained("./bert_localpath")

text = "This is an example sentence for text cleaning and tokenization."

# 清洗文本
cleaned_text = clean_text(text)

# 分词并转换为输入张量
inputs = tokenizer(cleaned_text, return_tensors='pt')

# 模型推理
with torch.no_grad():
    outputs = bert(**inputs)

# 获取最后一层的隐藏状态
last_hidden_states = outputs.last_hidden_state
print(last_hidden_states.shape)