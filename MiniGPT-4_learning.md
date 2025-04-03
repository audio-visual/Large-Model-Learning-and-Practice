学习案例原始地址：https://github.com/Vision-CAIR/MiniGPT-4/

为了简单起见，以minigpt4(llama)版本为例

前置实验，LLama的使用：https://github.com/audio-visual/Large-Model-Learning-and-Practice/blob/main/LLama_simple_usage.md

# 环境搭建（推理）
可参考原始工程
我做的改动是

1、先把models/eva-vit.py中vit的加载换成了本地路径

2、因为是在autodl的平台上跑，所以

gradio启动换成了autodl平台需要的端口 `gradio.lanch(server_port=6006)`

然后autodl服务器运行

`python demo.py --cfg-path eval_configs/minigpt4_llama2_eval.yaml  --gpu-id 0`

最后本地运行 （这个其实是远程服务器厂商做的端口映射）,记得替换用户名为自己的，端口也是
输入该命令后终端会提示输入租用服务器的密码
`ssh -CNg -L 6006:127.0.0.1:6006 username@connect.nmb1.seetacloud.com -p 24542`

即可在本地浏览器打开 127.0.0.1:6006

# 推理
首先，把模型当作黑箱，只考虑其对外提供的api功能： 一是visual_encoder，提供图像的处理与编码功能；二是LLM（实际上已经是多模态大模型），提供编码、输出功能。
那么推理部分实际上核心在于`Gradio`, `Chat`与`Conversation`

`Gradio` 提供webui接口，用户可在该界面上传图片，然后发送文本与大模型进行交互

`Chat`是实现Gradio对外按钮/功能的类，包括上传图片`upload_img()`、编码图片`encode_img()`、处理问题`gradio_ask()`、回答问题`gradio_answer()`

`Conversation`这个类是为了统一对message调整格式以及处理

而能够让整体的会话有记忆的关键在于`gradio`提供的`chat_state = gr.State()`,`img_list = gr.State()` (位于demo.py中)，它们相当于容器，可以存放变量，充当gradio与Chat之间的交流桥梁。
以及gradio框架提供的回调机制，是该容器能够充当桥梁的原因。

**重点功能1：上传编码图片**
```python
# demo.py
# 第一个list,函数upload_img的入参，第二个list是upload_img的返回值
upload_button.click(upload_img, [image, text_input, chat_state], [image, text_input, upload_button, chat_state, img_list])
```
    
```python
# demo.py
def upload_img(gr_img, text_input, chat_state):
    if gr_img is None:
        return None, None, gr.update(interactive=True), chat_state, None
  
    # copy感觉可以理解为，每次上传一张图，都会开启新的一轮对话
    chat_state = CONV_VISION.copy() 
    img_list = [] 
    llm_message = chat.upload_img(gr_img, chat_state, img_list)
    chat.encode_img(img_list)
    # 关键就在于，当前copy了的一次会话（chat_state），以及图像编码后结果（储存在了img_list）,会以参数返回的形式，被gradio中的`chat_state = gr.State()`,`img_list = gr.State()`接收，因此后续能继续针对该图片展开对话
    return gr.update(interactive=False), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list
```

```python
# conversation.py
class Chat:
    #....
    def upload_img(self, image, conv, img_list):
        conv.append_message(conv.roles[0], "<Img><ImageHere></Img>") # [['<s>[INST] ', '<Img><ImageHere></Img>']]
        img_list.append(image) 
        msg = "Received."
        return msg

    def encode_img(self, img_list):
        image = img_list[0]
        img_list.pop(0)
        if isinstance(image, str):  # is a image path
            raw_image = Image.open(image).convert('RGB')
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, Image.Image):
            raw_image = image
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)

        image_emb, _ = self.model.encode_img(image)
        img_list.append(image_emb)
```

**重点功能2：上传回答问题**
```python
# demo.py
# 顺序执行，先gradio_ask,结束后再gradio_answer
text_input.submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
        gradio_answer, [chatbot, chat_state, img_list, num_beams, temperature], [chatbot, chat_state, img_list]
    )
```

```python
# demo.py
def gradio_ask(user_message, chatbot, chat_state):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chat.ask(user_message, chat_state) # chat_state在该函数内进行更新
    chatbot = chatbot + [[user_message, None]] # 更新gradio对话框中的内容
    return '', chatbot, chat_state
```

```python
# conversation.py
class Chat:
    def ask(self, text, conv):
        ```
         [['<s>[INST] ', '<Img><ImageHere></Img> describe this image'], [' [/INST] ', 'The image is of a man standing in a field with a camera slung over his shoulder. He is wearing a brown t-shirt and black pants, and his hair is cut short. In the background, there are trees and hills, and the sky is clear with no clouds.'], ['<s>[INST] ', "what's the color of the background?"]]
        ```
        # 若是上传了新的图片，还没开始第一次对话时，则走该分支
        if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
                and conv.messages[-1][1][-6:] == '</Img>': 
            # ['<s>[INST] ', '<Img><ImageHere></Img> describe this image']
            conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])
        else:#否则，就是如下分支 # [INST] ', "what's the color of the background?"]
            conv.append_message(conv.roles[0], text)
```

```python
# demo.py
def gradio_answer(chatbot, chat_state, img_list, num_beams, temperature):
    # step1: answer_prepare
    # step2: model_generate
    # step3: model.llama_tokenizer.decode
    # step4: reformulate
    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=300,
                              max_length=2000)[0]
    chatbot[-1][1] = llm_message
    return chatbot, chat_state, img_list
```

```python
class Chat:
    def answer(self, conv, img_list, **kargs):
        generation_dict = self.answer_prepare(conv, img_list, **kargs)
        output_token = self.model_generate(**generation_dict)[0]
        output_text = self.model.llama_tokenizer.decode(output_token, skip_special_tokens=True)

        # 这儿和训练时候用的标准格式有关
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()

        conv.messages[-1][1] = output_text # [' [/INST] ', 'The image is of a man standing in a field with a camera slung over his shoulder. He is wearing a brown t-shirt and black pants, and his hair is cut short. In the background, there are trees and hills, and the sky is clear with no clouds.']
        return output_text, output_token.cpu().numpy()

    def answer_prepare(self, conv, img_list, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
                       repetition_penalty=1.05, length_penalty=1, temperature=1.0, max_length=2000):
        conv.append_message(conv.roles[1], None)# 消息结束
        # Give the following image: <Img>ImageContent</Img>. You will be able to see the image once I provide it to you. Please answer my questions.<s>[INST] <Img><ImageHere></Img> describe this image? [/INST]
        prompt = conv.get_prompt() #Conversation的结构化处理，系统命令+sep+用户消息
        # 根据prompt和已经编码后的图像，得到经过多模态模型编码的embedding
        embs = self.model.get_context_emb(prompt, img_list)

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)
        embs = embs[:, begin_idx:]

        generation_kwargs = dict(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=float(temperature),
        )
        return generation_kwargs

    def model_generate(self, *args, **kwargs):
        # for 8 bit and 16 bit compatibility
        with self.model.maybe_autocast():
            output = self.model.llama_model.generate(*args, **kwargs)
        return output
```

# 训练
TODO
