import os
import json
import random
from PIL import Image
import torch
import torch_npu
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from processor.preprocess import build_blip2_image_train_processor

def custom_collate_fn(batch):

    questions = [item['questions'] for item in batch]
    answers = [item['answers'] for item in batch]
    history_actions = [item['history_actions'] for item in batch]
    num_images = torch.tensor([item['num_images'] for item in batch])
    
    max_num_images = num_images.max().item()
    
    c, h, w = batch[0]['images'].shape[1:]
    
    padded_images = torch.zeros(len(batch), max_num_images, c, h, w)
    
    for i, item in enumerate(batch):
        seq_len = item['num_images']
        padded_images[i, :seq_len, :, :, :] = item['images']
        
    return {
        "images": padded_images,        
        "questions": questions,   
        "answers": answers,             
        "num_images": num_images, 
        "history_actions": history_actions,          
    }


class NavigationDataset(Dataset):
    def __init__(self, json_dir, image_root_dir, transform=None, history_frames=None):
        """
        Args:
            json_dir (str): 存放json文件的目录路径
            image_root_dir (str): 存放所有 ep_x_images 文件夹的根目录
            transform (callable, optional): 图像预处理 transforms
        """
        self.action_map = {
            0: "stop",
            1: "move forward",
            2: "turn left 15 degrees",
            3: "turn right 15 degrees",
        }
        self.json_dir = json_dir
        self.image_root_dir = image_root_dir
        self.history_frames = history_frames
        if transform is None:
            self.transform = build_blip2_image_train_processor()
        elif isinstance(transform, dict):
            self.transform = build_blip2_image_train_processor(cfg=transform)
        else:
            self.transform = transform

        self.json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]  # 数据集文件夹下除了traj_x.json以外任何.json文件都不能有
        
    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_filename = self.json_files[idx]
        json_path = os.path.join(self.json_dir, json_filename)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        episode_id = data['episode_id']
        instruction = data['instruction']
        steps = data['steps']

        n = len(steps)
        if n < 2:
            raise ValueError(f"Episode {episode_id} has only {n} step(s); need at least 2.")
        if self.history_frames is None:
            if n < 3:
                t = n
            else:
                t = random.randint(3, n) # num_images > 2
            selected_steps = steps[:t]
        else:
            try:
                history_frames = int(self.history_frames)
            except (TypeError, ValueError):
                history_frames = None
            if history_frames is None:
                if n < 3:
                    t = n
                else:
                    t = random.randint(3, n) # num_images > 2
                selected_steps = steps[:t]
            else:
                if history_frames < 1:
                    history_frames = 1
                history_frames = min(history_frames, n - 1)
                total_frames = history_frames + 1
                max_start = n - total_frames
                start_idx = 0 if max_start <= 0 else random.randint(0, max_start)
                selected_steps = steps[start_idx:start_idx + total_frames]
                t = len(selected_steps)

        image_tensors = []
        image_folder_name = f"ep_{episode_id}_images"
        episode_image_dir = os.path.join(self.image_root_dir, image_folder_name)

        for step_data in selected_steps:
            image_name = step_data['image_path']
            image_path = os.path.join(episode_image_dir, image_name)
            
            try:
                image = Image.open(image_path).convert('RGB')
            except FileNotFoundError:
                print(f"Warning: Image not found {image_path}")
                image = Image.new("RGB", (224, 224))

            if self.transform:
                image = self.transform(image)
            image_tensors.append(image)

        if isinstance(image_tensors[0], torch.Tensor):
            images_sequence = torch.stack(image_tensors) 
        else:
            to_tensor = transforms.ToTensor()
            images_sequence = torch.stack([to_tensor(img) for img in image_tensors])
        
        history_actions = [self.action_map[step['action']] for step in selected_steps[:-1]]
        last_step_data = selected_steps[-1]
        target_action_id = last_step_data['action']
        target_action = self.action_map[target_action_id]
        samples={
            "images": images_sequence, # Shape: [n, C, H, W]
            "num_images": t,
            "answers": target_action,
            "questions": instruction,
            "history_actions": history_actions,
        }
        return samples