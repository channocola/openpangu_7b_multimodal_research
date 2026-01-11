"""
    copyright: Uestc-ChenQi
    请联系： chan.nocola@gmail.com
"""

### 2026/1/1 更新：

修复了pgmm训练时，未在text末尾加入<eos>的bug：
    若训练文本中未加入 <eos> 标记，会导致 Q-Former 无法学习到用于指示 LLM 停止生成的潜在特征。尽管 LLM 处于冻结状态，但它仍依赖 Query Tokens 提供的上下文特征（Contextual Cues）来判断何时终止输出。缺少 <eos> 会导致 Query Token 缺失这一关键的终止偏置信号；
    得益于pgmm中对应token的label处理，没有<eos>不影响之前图文对齐（即Stage2 Pretrain，task（lavis）：Image-Text Pretrain ，实际逻辑为caption）训练结果；若此时已有没有<eos>的ckpt，请load全量ckpt并修改max_epoch参数（建议修改为3-5），继续开始训练，训练开始时损失增大是正常的，因为没有<eos>的训练其实会导致训练损失虚高
    ***bug修改在main分支的pgmm.py中完成，请重新clone或pull main分支***

重要更新：引入deepspeed大模型训练框架：
    请在pgmm训练环境的基础上，安装deepspeed-0.18.3，请使用如下命令行：
    ***pip install deepspeed=0.18.3***
    目前deepspeed可以无缝衔接npu，请不要再尝试安装对应版本号的deepspeed_npu

重要更新：直接引入LoRA和Adapter微调方法，不借助任何其它第三方工具：
    可以在 WorkSpace/pgn/llm/configuration_openpangu_dense.py 中直接启用/禁用这两种微调方式

### VLN:Visiual-Language-Navigation system --- pgn：pangu navigation 项目代码结构与说明：

本盘古视觉语言导航模型是基于华为公司开发的openpangu_7b大语言模型开发，其多模态能力基于开源lavis库进行开发，视觉语言导航能力属多模态下游任务，开发时已与lavis库解耦，整体模型forward逻辑、训练任务及训练方法均属创新
模型基本结构为：

image--------->Vision_tower(eva-vit-g)<freeze> <or PEFT>                       
                |
                |----Query_module(q_former)<training>             text-------> openpangu7b_tokeninzer<freeze but extend>
                |                                                               |
                |----mm_mlp_projector(2x_glue + Linear)<training>               |----openpangu7b_embedding<freeze but extend>
                |                                                               |
                |[vision_token/query_token]                                     |[text_token]
                |                                                               |
                \/                                                              \/
                ---------------------LLM(openpangu_7b)<PEFT>---------------------       

#### 代码基本结构

##### 多模态盘古模型基本结构模型代码

盘古视觉语言导航模型被命名为pgn，整体模型代码与上一阶段预训练不同，与lavis库解耦，代码独立
***重要说明：尽管当前版本仅预置了 VLN（视觉语言导航）任务，但本框架采用模块化设计，已为 Image Captioning、VQA 等多模态下游任务预留了通用接口。扩展新任务时无需重构核心代码，仅需通过注册机制（Registry Mechanism）配置相应的任务（Task）、数据处理器（Processor）和数据集（Dataset）即可实现。***

pgn模型代码可在：/WorkSpace/pgn中被找到
其中：/WorkSpace/pgn
                |
                |----------configs  ***配置文件文件夹***
                |           |----------deepspeed_zero2.json   ***ZeRO-2stage 配置文件，若能有现存紧张情况，请在此处更改stage为3***
                |           |----------navigation_train_base_on_mmpangutrain.yaml   ***pgn训练配置，所有训练参数均应在此处被定义***
                |----------llm   ***为openpangu_7b模型结构代码文件夹***
                |           |----------configuration_openpangu_dense.py   ***openpangu_7b模型配置，请在此处设置PEFT方法，目前支持LoRA与Adapter***
                |           |----------***其它openpangu_7b模型结构代码***
                |----------processor  ***为下游任务准备的数据处理器文件夹***
                |           |----------base_processor.py  ***基本预处理器***
                |           |----------preprocess.py   ***VLN专用预处理，只进行图像数据的归一化***
                |----------base_dataset.py   ***基本数据集***
                |----------dataset.py  ***VLN专用数据集***
                |----------ntrain.py   ***VLN专用训练入口***
                |----------pgn.py    ***openpangu_7b-VLN标准模型定义，定义VLN前向传播逻辑***
                |----------register.py   ***为通用化适配下游任务的注册器（占位）***
                |----------runner.py   ***适配各类下游任务的通用化启动器，可进行训练、推理和测试***
                |----------q_former   ***包含vision_tower/q-former/projector以及各类工具的文件夹***
                            |----------eva_vit.py  ***vision_tower***
                            |----------blip2.py  ***模型基类***
                            |----------pgnconfig.py   ***pgn模型本身配置文件***
                            |----------projector.py   ***pgn模型视觉（query_token）投影层***
                            |----------q_former.py   ***pgn模型视觉前端整合***
                            |----------Qformer.py   ***q-former标准定义***
                            |----------***其它为本模型适配的通用工具，如logger、分布式数据采样器等***

##### pgn模型训练时所用代码

所有训练都对昇腾npu进行了适配

本次训练由于数据量大，序列长，使用deepspeed ZeRO-2进行分布式训练，将lavis中DDP移除
训练启动命令：
    ***deepspeed --num_gpus 8 ntrain.py --cfg-path configs/navigation_train_base_on_mmpangutrain.yaml***
    （num_gpu不影响训练启动）

VLN训练通过专用训练入口启动：ntrain.py
ntrain.py调用启动器runner.py，如果是VLN训练，则不需要给启动器传任何"task"字段，启动器将会自动使用default_training_loop，使用pgn内部前向传播逻辑开始训练，使用默认的dataset.py组织数据，使用默认的preprocessor.py进行数据处理。

请根据实际硬件情况及实际需求修改以下两项配置文件：
    ***deepspeed_zero2.json***
    ***navigation_train_base_on_mmpangutrain.yaml***
其中，***navigation_train_base_on_mmpangutrain.yaml***中定义了训练epoch数、学习率调度方法、数据路径等关键训练参数

请根据实际需求修改以下配置文件：
    ***configuration_openpangu_dense.py***
***注意！请只修改其中的use_adapter与use_lora两项参数选择你需要使用的微调方式，其他参数不建议改动，不建议同时开启adapter与lora***

如果使用者希望使用本模型继续训练VLM下游任务（即基于pgmm多模态预训练权重继续微调训练），但不希望仅进行VLN训练，可在register.py内部注册一个新的具体的任务（至少包括损失定义、评估标准与指标等），定义一个新的dataset用于新任务的专用数据集，基于base_processor.py定义新的数据处理器进行符合任务需求的数据预处理，重写pgn的前向传播逻辑并修改名称为pgxxx.py，对ntrain.py进行针对性改动并修改名称为xxxtrain.py，训练时把定义的新任务传入启动器runner并使用deepspeed在昇腾计算卡上进行训练即可

### pgn模型前向传播逻辑与数据集逻辑<VLN训练与推理逻辑>

训练数据：使用本作者github账号上发布的项目另一个"navid模拟数据生成"（***注意：该分支只能在gpu上运行***）分支中生成的模拟数据进行训练
    训练时所用数据可在 ***/WorkSpace/pgn/ep*** 中找到示例，一般情况下，一个回合（或一个场景）对应一个json文件和一个图像文件夹下的所有图片
    如： ***/WorkSpace/pgn/ep/traj_5.json*** 与 ***/WorkSpace/pgn/ep/ep_5_images***

#### 简要一般VLN数据集逻辑（单个样本）：

识别回合id并找到对应回合的图像文件夹路径，在json文件中提取 ***instruction*** 作为当前回合下的 ***question*** ，识别整个回合的实际帧数n，生成随机数 ***t***（***注意，t∈[0,n-1]***），以第t帧为起点，向后数 ***history_frames*** 帧，并以最后一帧作为当前帧（***注意，history_frames可在/WorkSpace/pgn/configs/navigation_train_base_on_mmpangutrain.yaml中进行修改，如果不传该参数，逻辑为，生成随机数 t ，t∈[3,n-1]，以t为终点（当前帧），以第0帧为起点提取子序列进行当前帧行动预测，请根据实际显存大小进行调整history_frames。另外，在没有设置历史帧数的情况下，batch_size不为1时可能会导致样本之间序列不等长，用黑图进行padding***），提取每一帧的 ***action***，并以最后一帧的action作为 ***answer***。

#### 简要一般VLN模型pgn前向传播逻辑：

扩充tokenizer词表，扩充词表embedding矩阵（***注意：请在cpu上进行词表扩展，否则极易OOM***），扩充：<|image_start|><|image_end|><|frame_sep|>,并在启动器runner扩展openpangu_7b的权重张量（***若没有扩展，则在训练加载权重时时会直接报错：张量不匹配，请在cpu上进行权重扩展***）
最终token组织形式：
    ***cur_text = f"{self.prompt0}\n{cur_question}\n{self.prompt2}\n{image_start}{sep_total}{image_end}\n{self.prompt1}\n{image_start}{image_end}\n{self.prompt_answer}\n{cur_answer}{eos_token_str}"***
    注意：openpangu_7b会自动加入<bos>，但不会自动加入<eos>
        
    自定义prompt如下:
        prompt0 = "You are a helpful navigation assistant. Based on what you see, please output the action you should take. Your output should only include: move forward, turn left 15 degrees, turn right 15 degrees, or stop.",
        prompt2 = "These are the scenes you saw and the actions you took:",
        prompt1 = "The last picture you see is what you see from your current location.",
        prompt_action = "action taken:",
        prompt_answer = "please take your action now:",

***关于历史帧的额外说明：***
需要历史帧和历史帧对应动作除了使模型能学会处理时序信息、使模型学会如何输出连续的action（运动是时空连续的）之外，还需要模型能基于LLM的启发性学会 ***搜素方法*** ，在模型没有明确看到终点时，需要持续左转/右转去搜索可能的终点，为了避免模型 ***迷茫的随机转动***，需要加入历史搜索逻辑，即：当历史帧保持左转/右转，且仍然没有找到可能的目标时，需要保持历史帧的搜索，继续左转/右转，这样就能使模型从历史步骤中学会如何在当前视野下做出合理动作规划，从而缓解整体模型的随机性

***sep_total额外说明：***
此处{sep_total} = <image> + <prompt_action> + <history_action_i> + <|frame_sep|> + <image> + <prompt_action> + <history_action_i+1> + .....

请根据实际硬件设备情况严格控制序列长度，避免OOM！

#### 加载权重方式：

可以在navigation_train_base_on_mmpangutrain.yaml中进行设置，选择整包ckpt加载，或是选择断点续训加载，或是自加载，优先级如下：
    ***resume_ckpt_path: null > pretrained_ckpt_path(一般指我们前一阶段pgmm的预训练结果) >   load_llm_pretrained: False + load_vision_pretrained: False + load_qformer_pretrained: False***

***注意：这种使用生成数据+随机数+vqa形式训练是对标准具身智能VLN任务训练的模仿，即利用大量数据并引入随即量模拟模型遇到的各种情景，此做法虽能达到一定效果，但其实最大瑕疵在于默认模型历史帧所有action全部正确，没有误差累积，这导致最大的后果是模型没有学会纠错，也没有对应的纠错机制；本作者认为标准VLN训练仍需要使用DAgger进行RL，即强化学习，模型在一个回合内逐步预测action，与专家轨迹进行比对，并加入误差累积，让模型学会自我纠错***


### 训练复现

#### 环境配置

支持openpangu_7b的环境配置：
操作系统：Linux（推荐 openEuler>=24.03）
python==3.11
固件驱动 >= 23.0.6
CANN: 8.1.rc1
vllm: 0.9.2
vllm-ascend: 0.9.2rc1
torch: 2.5.1
torch_npu: 2.5.1.post1
opencompass: 0.5.0
transformers: 4.53.2

支持lavis库的环境配置（已经删除一些不必要的库）：
contexttimer
einops>=0.4.1
fairscale==0.4.4
ftfy
iopath
ipython
omegaconf
opencv-python-headless==4.5.5.64
opendatasets
packaging
pandas
plotly
pre-commit
pycocoevalcap
pycocotools
python-magic
scikit-image
sentencepiece
spacy
streamlit
timm==0.4.12
torchvision
tqdm
webdataset
wheel

支持大模型分布式训练：
deepspeed=0.18.3

#### 训练数据

请在： ***/root/WorkSpace/pgn/configs/navigation_train_base_on_mmpangutrain.yaml*** 处修改json文件和对应图像数据文件的存储地址，建议将二者存放到同一路径下：
数据组织形式：
        path/to/your/training_data-----
                                    |-----------------------------traj_1.json
                                    |-----------------------------traj_2.json
                                    |-----------------------------traj_3.json
                                    .........
                                    .........
                                    |-----------------------------ep_1_images
                                    |-----------------------------ep_2_images
                                    |-----------------------------ep_3_images

VLN数据集由本作者生成，若需要取得完整6w+条数据，请联系本作者：chan.nocola@gmail.com
或您也可以在gpu服务器上配置好本项目另一个分支"navid模拟数据生成"，自行生成所需要的数据量
本作者稍后会将pgmm预训练权重和完整VLN数据更新到同一下载路径

#### 注意事项：

训练时checkpoint储存路径：
    在总训练配置 ***/root/WorkSpace/pgn/configs/navigation_train_base_on_mmpangutrain.yaml***
    的output_dir: "/path/to/your/blip2pangu_navigation_checkpoint" 处进行修改，换成需要的配置路径

训练时llm路径：
    在整体多模态模型配置处进行修改：***/WorkSpace/pgn/q_former/pgnconfig.py***
    修改位置：llm_path: str = "/path/to/your/pangu_model"
        ----补充说明：建议在此处放上完整的openpangu_7b代码仓库（至少包括openpangu_7b的四个权重文件和权重索引文件model.safetensors.index.json），下载路径为：
        https://ai.gitcode.com/ascend-tribe/openpangu-embedded-7b-model/?utm_source=gitcode_aigc_v1_t0&index=top&type=card&&uuid_tt_dd=10_36701610040-1757313585050-960456&isLogin=1&from_id=150223483&from_link=3187faac3a207ab24f41b173e4f60127

训练时预训练权重加载：
    应基于第一阶段pgmm预训练权重继续训练
    第一阶段训练权重稍后开源，本作者届时将直接更新一个新的分支存放本作者的预训练权重（大约42.34G）
    请在总训练配置 ***/root/WorkSpace/pgn/configs/navigation_train_base_on_mmpangutrain.yaml***
    的model:
        pretrained_ckpt_path: "/path/to/blip2pangu_checkpoint/20260104205/checkpoint_9.pth"处修改为您的路径


***注意，以下这段预训练权重加载注意事项与pgmm的相同，对应本阶段微调训练时自加载预训练权重逻辑***
由于下载代码后路径发生了变化，所以在复现训练时需要针对一些路径做出一些变化，如：
----总训练配置 ***/root/WorkSpace/pgn/configs/navigation_train_base_on_mmpangutrain.yaml***
    中定义的  pretrained: "/data1/blip2_pretrained/blip2_pretrained.pth"  即需要加载的预训练（此处的预训练指的是blip2_qformer的预训练权重，不带大语言模型解码器）
    ----另外，该预训练权重可能会因为服务器网络问题无法下载或下载速度很慢，建议到官网下载在上传到服务器的指定地址。
    在加载该预训练权重时会报出一些键对不匹配的情况，这是正常的，请忽略掉相关信息。
    Vision_tower的权重在eva_vit.py中有自定义加载，且在第一次训练时会看到在下载eva-vit-g的权重，llm的权重是我们利用huggingface的frompretrained方法自动加载的（此处额外说明，加载的该预训练权重本身就不应该报出loss： llm_keys相关的问题，因为该预训练权重本身就不应该包括llm的权重，此处报出该信息应该是与lavis本身定义的加载权重时的检查机制有关，但无论如何，这个报loss信息完全可以忽略），投影层是我们新定义参与训练的，故整体模型对于该预训练权重只需要加载q_former的权重即可
        ----补充信息：预训练权重下载地址：https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth 



#### 适配工作

##### 环境、算子等代码适配工作
注意，本下游任务训练虽已经与lavis库解耦，但仍需要lavis库环境配置：

昇腾计算设备与其环境生态已较为完善，但仍有一些库在昇腾计算设备上不被支持无法安装，如decord库，该库在lavis库中被用于加载视频数据，但由于无法安装，我们选择使用ffmeg绕过安装，并在训练中不考虑使用视频数据进行训练，只针对captioning任务进行训练而非vqa

另外，在模型训练过程中，我们发现在昇腾计算设备上启用AMP自动混合精度训练时会出现梯度爆炸。经排查，该问题主要源于Attention模块中的掩码操作在fp16半精度下引发的数值不稳定性，Mask操作引入的大幅值负数超出了fp16的范围，导致 Softmax 算子计算溢出。为确保训练的收敛性与稳定性，我们最终决定放弃AMP，转而采用全精度fp32进行训练计算。由于fp32相比混合精度显著增加了显存占用，作为平衡，我们在训练配置中相应调小了 batch_size 以适应硬件显存限制。

在模型 Checkpoint 保存及指标评估阶段，我们发现在昇腾的分布式通信后端HCCL中，部分集合通信算子（如 all_reduce）尚未原生支持 float64（Double）数据类型。若在多卡同步过程中直接传输双精度张量，会触发 RuntimeError 导致训练中断。
针对此硬件环境限制，我们在代码中实施了类型强制转换策略：在执行任何分布式同步操作或模型保存前，显式地将所有相关的标量和张量转换为 float32 类型，以确保与HCCL通信后端的兼容性，从而保证模型能够正常保存与断点续训。

***请注意！***
***请注意！***
***请注意！***
***由于昇腾npu目前不支持cholesky分解，同时也为了避免OOM，词表扩展，embedding扩展和openpangu_7b的权重扩展一定要在cpu上进行，之后再利用deepspeed移动到npu上，此举会拖慢一些训练启动的速度***

##### torch_npu适配工作
lavis 库原生设计面向 GPU（CUDA）环境，其部分训练相关的工具函数、API 接口及底层依赖均基于 CUDA 实现。为实现对昇腾 NPU 平台的适配，本文对训练过程中涉及的全部设备调用、算子接口及依赖库进行了逐项排查与重构，替换为兼容 NPU 的实现方案，从而保障模型在昇腾计算设备上的正确性与训练稳定性。


#### 开始训练复现

在完成上述注意事项之后，就可以进行训练复现了：

----配置openpangu_7b环境
----配置VLN环境
----使用训练入口：ntrain.py：
        命令行输入：***deepspeed --num_gpus 8 ntrain.py --cfg-path configs/navigation_train_base_on_mmpangutrain.yaml***   以支持八卡训练
