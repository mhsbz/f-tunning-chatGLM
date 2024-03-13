import time

import torch
from transformers import BertModel, BertTokenizer

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载预训练的BERT模型和分词器
model = BertModel.from_pretrained('bert-base-uncased').cuda()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = model.to(device)  # 将模型移动到选定的设备

# 准备输入数据
text = "Here is some text to encode."
encoded_input = tokenizer(text, return_tensors='pt')  # 将文本编码为模型输入
encoded_input = {k: v.to(device) for k, v in encoded_input.items()}  # 将输入数据移动到选定的设备

# 进行推理
with torch.no_grad():  # 关闭自动梯度计算以减少内存消耗
    output = model(**encoded_input)

# 获取最后一个隐藏层的输出
last_hidden_state = output.last_hidden_state



print(last_hidden_state)


time.sleep(30)

# 如果您需要进行进一步的下游任务，可以在这里处理输出
# 例如，进行文本分类任务，您可能需要使用last_hidden_state来获取分类结果
