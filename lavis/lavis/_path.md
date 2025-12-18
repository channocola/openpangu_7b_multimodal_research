tasks: 任务分类     ----PGN\multimodel\lavis\lavis\tasks\image_text_pretrain.py
runners： 训练、推理启动器   ----PGN\multimodel\lavis\lavis\runners\runner_base.py
projects： 存放训练、推理的配置----PGN\multimodel\lavis\lavis\projects\blip2\train\stage2_pretrain_openpangu7b.yaml
processors： 预处理  ----blip_processors.py 

datasets_builder:     ----PGN\multimodel\lavis\lavis\datasets\datasets\image_text_pair_datasets.py
                      ----PGN\multimodel\lavis\lavis\datasets\builders\caption_builder.py

datasets:    ----PGN\multimodel\lavis\lavis\datasets\datasets\image_text_pair_datasets.py
             ----PGN\multimodel\lavis\lavis\datasets\datasets\coco_caption_datasets.py
 
configs:    ----PGN\multimodel\lavis\lavis\configs\models\blip2\blip2_pretrain_openpangu7b.yaml