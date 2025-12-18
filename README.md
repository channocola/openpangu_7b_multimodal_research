"""
    copyright: Uestc-ChenQi
"""

### 代码结构与说明：

#### 模型基本结构
本多模态-盘古模型是基于华为公司开发的openpangu_7b大语言模型开发，其多模态能力基于开源lavis库进行开发
模型基本结构为：

image--------->Vision_tower(eva-vit-g)<freeze>                        
                |
                |----Query_module(q_former)<training>             text-------> openpangu7b_tokeninzer<freeze>
                |                                                               |
                |----mm_mlp_projector(2x_glue + Linear)<training>               |----openpangu7b_embedding<freeze>
                |                                                               |
                |[vision_token/query_token]                                     |[text_token]
                |                                                               |
                \/                                                              \/
                ---------------------LLM(openpangu_7b)<freeze>---------------------         

#### 代码基本结构

##### 多模态盘古模型基本结构模型代码
多模态盘古被命名为pgmm(pangu_multimodal) 整体模型代码被注册到lavis库中

pgmm模型代码可在：/lavis/lavis/models/blip2pangu_models 中被找到
其中： /lavis/lavis/models/blip2pangu_models
                            |
                            |----------modeling_openpangu_dense.py  为openpangu_7b模型结构代码，在pgmm.py中被调用
                            |----------modular_openpangu_dense.py   为openpangu_7b模型结构代码，在pgmm.py中被调用
                            |----------pgmm_qformer.py              为pgmm专用视觉结构代码，针对openpangu_7b做了适配，定义了pgmm所用整体视觉模块（包括Vision_tower和Query_module），在pgmm.py中被调用
                            |----------pgmm.py                      ***pgmm整体模型结构代码，整体多模态盘古模型在该脚本中被定义***
                            |----------pgmmconfig.py                为pgmm专用配置，定义了整体多模态模型的配置与多模态mlp投影层配置
                            |----------projector.py                 为pgmm专用mm_mlp_projector模块结构代码，针对openpangu_7b做了适配，定义了pgmm所用视觉投影层，在pgmm.py中被调用
                            |----------tokenization_openpangu.py    为openpangu_7b的tokenizer模型结构代码，在pgmm.py中被调用

另外，Vision_tower所用eva-vit-g模型在：/lavis/lavis/models/eva_vit.py 中被定义，在pgmm_qformer.py中被调用

##### 多模态盘古模型训练时所用代码
得益于lavis库的完善性，多模态盘古在训练时利用了lavis库中的训练工具

训练入口在/PGN/mmtrain.py 中被定义，该训练脚本针对openpangu_7b与昇腾计算设备做了针对性适配
mmtrain.py调用了/lavis/lavis/runners/runner_base.py 中定义的启动器来实现训练，启动器调用了dataloader、prepossessor等工具
其中，基本的base_dataloader在 /lavis/lavis/datasets/builders/base_dataset_builder.py 中被定义
        coco所用dataloader在 /lavis/lavis/datasets/builders/caption_builder.py 
                             /lavis/lavis/datasets/datasets/coco_caption_datasets.py 中被定义
        vg所用dataloader在 /lavis/lavis/datasets/builders/image_text_pair_builder.py 
                           /lavis/lavis/datasets/datasets/image_text_pair_datasets.py 中被定义

另外，上述脚本文件与相关的utils组件文件均针对openpangu_7b与昇腾计算设备做了适配

#### 配置文件基本结构

##### 多模态盘古模型训练所用配置文件
多模态盘古模型在训练时所用配置文件使用lavis库的注册机制被注册到lavis库中

训练时所用总训练配置：
总训练配置两份（两份相同），路径分别是：/PGN/projects/train/stage2_pretrain_openpangu7b.yaml
                                    /lavis/lavis/projects/blip2pangu/train/stage2_pretrain_openpangu7b.yaml
总训练配置中定义了所使用模型结构、模型类型、是否加载预训练模型、预训练模型路径、是否冻结vit、所用数据集、所用数据处理器、所用data_loader以及如epoch数、学习率、所用训练设备等一些常规训练配置
该总训练配置针对openpangu_7b模型做了适配

训练时所用数据集配置：
模型训练时所用数据集有两种：coco与vg（VisualGenome）
其中，coco数据集配置路径为：/lavis/lavis/configs/datasets/coco/defaults_cap.yaml
其中定义了数据集名称、数据集下载地址、本地数据组织格式等。另外需要注意的一点是：coco_karpathy_train.json与coco官方发布的coco_train.json不同，其对图像-文本对的数据组织格式是针对多模态训练重新组织过的，故图像数据可在coco官网下载，但是.json文件需要按照本配置文件的指示下载
vg数据集配置路径为：/lavis/lavis/configs/datasets/vg/defaults_caption.yaml
其中定义内容与注意事项与coco数据集相同

##### 预训练多模态盘古模型配置文件
预训练好的多模态盘古模型配置文件路径为：/lavis/lavis/configs/models/blip2/blip2_pretrain_openpangu7b.yaml
该配置文件是为了适配lavis库而存在的，应当是调用pgmm进行推理时，加载pgmm所用配置文件，但该配置文件中大部配置（除指向预训练权重路径的pretrained与指向所用预处理器）与pgmmconfig.py重合

#### 适配工作

##### 环境、算子等代码适配
昇腾计算设备与其环境生态已较为完善，但仍有一些库在昇腾计算设备上不被支持无法安装，如decord库，该库在lavis库中被用于加载视频数据，但由于无法安装，我们选择使用ffmeg绕过安装，并在训练中不考虑使用视频数据进行训练，只针对captioning任务进行训练而非vqa

另外，在模型训练过程中，我们发现在昇腾计算设备上启用AMP自动混合精度训练时会出现梯度爆炸。经排查，该问题主要源于Attention模块中的掩码操作在fp16半精度下引发的数值不稳定性，Mask操作引入的大幅值负数超出了fp16的范围，导致 Softmax 算子计算溢出。为确保训练的收敛性与稳定性，我们最终决定放弃AMP，转而采用全精度fp32进行训练计算。由于fp32相比混合精度显著增加了显存占用，作为平衡，我们在训练配置中相应调小了 batch_size 以适应硬件显存限制。

在模型 Checkpoint 保存及指标评估阶段，我们发现在昇腾的分布式通信后端HCCL中，部分集合通信算子（如 all_reduce）尚未原生支持 float64（Double）数据类型。若在多卡同步过程中直接传输双精度张量，会触发 RuntimeError 导致训练中断。
针对此硬件环境限制，我们在代码中实施了类型强制转换策略：在执行任何分布式同步操作或模型保存前，显式地将所有相关的标量和张量转换为 float32 类型，以确保与HCCL通信后端的兼容性，从而保证模型能够正常保存与断点续训。

##### torch_npu适配工作
lavis 库原生设计面向 GPU（CUDA）环境，其部分训练相关的工具函数、API 接口及底层依赖均基于 CUDA 实现。为实现对昇腾 NPU 平台的适配，本文对训练过程中涉及的全部设备调用、算子接口及依赖库进行了逐项排查与重构，替换为兼容 NPU 的实现方案，从而保障模型在昇腾计算设备上的正确性与训练稳定性。

### 训练复现

#### 复现pgmm训练的注意事项

由于下载代码后路径发生了变化，所以在复现训练时需要针对一些路径做出一些变化，如：
----总训练配置/PGN/projects/train/stage2_pretrain_openpangu7b.yaml
             /lavis/lavis/projects/blip2pangu/train/stage2_pretrain_openpangu7b.yaml
    中定义的  pretrained: "/data1/blip2_pretrained/blip2_pretrained.pth"  即需要加载的预训练（此处的预训练指的是blip2_qformer的预训练权重，不带大语言模型解码器）权重地址
    ----另外，该预训练权重可能会因为服务器网络问题无法下载或下载速度很慢，建议到官网下载在上传到服务器的指定地址，在加载该预训练权重时会报出一些键对不匹配的情况，这是正常的，Vision_tower的权重在eva_vit.py中有自定义加载，且在第一次训练时会看到在下载eva-vit-g的权重，llm的权重是我们利用huggingface的frompretrained方法自动加载的（此处额外说明，加载的该预训练权重本身就不应该报出loss： llm_keys相关的问题，因为该预训练权重本身就不应该包括llm的权重，此处报出该信息应该是与lavis本身定义的加载权重时的检查机制有关，但无论如何，这个报loss信息完全可以忽略），投影层是我们新定义参与训练的，故整体模型对于该预训练权重只需要加载q_former的权重即可
        ----补充信息：预训练权重下载地址：https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth 

----训练时数据集路径配置：
    /lavis/lavis/configs/default.yaml 在此处修改存放数据集的具体路径即可
        ----补充信息：数据集组织格式：----path_to_your_dataset
                                      |
                                      |----vg------------
                                      |                 |                  
                                      |                 |----------annotations -------
                                      |                 |                           |-----------------vg_caption.json
                                      |                 |    
                                      |                 |----------images-------------
                                      |                                             |-----------------VG_100K  
                                      |                                             |-----------------VG_100K_2
                                      |
                                      |----coco----------
                                                        |
                                                        |----------annotations -------
                                                        |                           |-----------------captions_train2014.json  coco_karpathy_test.json   coco_karpathy_val.json    instances_val2014.json       
                                                        |                                             captions_val2014.json    coco_karpathy_train.json  instances_train2014.json  person_keypoints_train2014.json
                                                        |
                                                        |
                                                        |----------images-------------
                                                                                    |
                                                                                    |-----------------test2014  test2015 train2014  val2014

----训练时checkpoint储存路径：
    在总训练配置/PGN/projects/train/stage2_pretrain_openpangu7b.yaml
             /lavis/lavis/projects/blip2pangu/train/stage2_pretrain_openpangu7b.yaml
    的output_dir: "/data1/blip2pangu_checkpoint"处进行修改，换成需要的配置路径

----训练时llm路径：
    在整体多模态配置处进行修改：/lavis/lavis/models/blip2pangu_models/pgmmconfig.py
    修改位置：llm_path: str = "/data1/pangu_model"
        ----补充说明：建议在此处放上完整的openpangu_7b代码仓库（至少包括openpangu_7b的四个权重文件和权重索引文件model.safetensors.index.json），下载路径为：https://ai.gitcode.com/ascend-tribe/openpangu-embedded-7b-model/?utm_source=gitcode_aigc_v1_t0&index=top&type=card&&uuid_tt_dd=10_36701610040-1757313585050-960456&isLogin=1&from_id=150223483&from_link=3187faac3a207ab24f41b173e4f60127


#### 复现pgmm训练时可能会遇到的问题

可能会遇到环境配置等问题，pgmm需要的环境配置如下：

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

#### 训练复现

在完成上述注意事项之后，就可以进行训练复现了：

----配置openpangu_7b环境
----配置lavis库环境并进行安装
----使用训练入口：mmtrain.py：
        命令行输入：python -m torch.distributed.run --nproc_per_node=8 mmtrain.py --cfg-path /PATH/TO/YOUR/PRETRAIN.YAML   以支持八卡训练





