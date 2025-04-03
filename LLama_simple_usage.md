这是MiniGPT-4的前置实验，主要是确认一下LLama模型本身是否能够加载成功以及正常推理

单卡版本
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
device = 'cuda:2'
# 很奇怪，明明关于与模型，safetensors和bin只需要任意一种，但是只下载bin，会报错，说缺safetensors
def load_model(local_file=None):
    if torch.cuda.is_available():
        print("OK cuda is available!")
        if local_file:
            model = AutoModelForCausalLM.from_pretrained(local_file, torch_dtype=torch.float16)
            tokenizer = AutoTokenizer.from_pretrained(local_file)
        else:
            model_id = "meta-llama/Llama-2-7b-chat-hf"
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        model.to(device)
        tokenizer.use_default_system_prompt = False
        
        return model, tokenizer
model, tokenizer = load_model(local_file="projects/weights/Llama-2-7b-chat-hf") # 本地路径

def chat_with_llama(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to(device)
    output = model.generate(input_ids, max_length=256, num_beams=4, no_repeat_ngram_size=2)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

while True:
    prompt = input("You: ")
    response = chat_with_llama(prompt)
    print("Llama:", response)
```

测试了一下多卡版本，发现速度确实能快不少
```python
# 测试下模型的并行
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoTokenizer,AutoModelForCausalLM

model_path = "projects/weights/Llama-2-7b-chat-hf"

# 1. 初始化空模型
with init_empty_weights():
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_config(config)

# 2. 自动分配模型到可用设备
model = load_checkpoint_and_dispatch(
    model,
    model_path,
    device_map="auto",  # 自动并行
    no_split_module_classes=["LlamaDecoderLayer"],  # 指定不拆分的模块
    dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_path)


def chat_with_llama(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to('cuda')
    output = model.generate(input_ids, max_length=256, num_beams=4, no_repeat_ngram_size=2)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

while True:
    prompt = input("You: ")
    response = chat_with_llama(prompt)
    print("Llama:", response)
```
