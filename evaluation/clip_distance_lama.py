import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import json
from tqdm.auto import tqdm
import pandas as pd
import argparse
import clip
import csv
import glob

class CLIPMetric:
    def __init__(self, model_name="ViT-B/32", device="cuda"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        model, preprocess = clip.load(model_name, device=self.device)
        self.model = model.eval()
        self.preprocess = preprocess

    def score(self, images, texts):
        images = images.to(self.device)
        
        # 确保texts是一个list
        if not isinstance(texts, list):
            texts = [texts]

        scores = []
        for img, text in zip(images, texts):
            text_tokenized = clip.tokenize(text).to(self.device)
            img = img.unsqueeze(0).to(self.device)  # 添加批次维度

            with torch.no_grad():
                logits_per_image, logits_per_text = self.model(img, text_tokenized)

            scores.append(logits_per_image.squeeze().cpu().numpy())
        # 将每个NumPy数组转换为PyTorch张量
        tensor_list = [torch.tensor(arr) for arr in scores]

        # 将这些张量堆叠为一个新的张量
        stacked_tensor = torch.stack(tensor_list)

        # 调整形状为[4, 1]
        final_tensor = stacked_tensor.unsqueeze(1)
        return final_tensor
    
    def calculate_clip_consensus(self, images):  
        std = []
        for img_seed1,img_seed2,img_seed3 in zip(*images):
            img_seeds = [img_seed1, img_seed2, img_seed3]
            embeddings = []
            for image in img_seeds:
                image = image.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    image_features = self.model.encode_image(image)
                embeddings.append(image_features.cpu().numpy())

            # 将嵌入堆叠成一个numpy数组
            embeddings = np.vstack(embeddings)
            
            # 计算嵌入的标准差
            consensus_std = np.std(embeddings, axis=0)
            std.append(consensus_std.mean())
        # 将每个NumPy数组转换为PyTorch张量
        tensor_list = [torch.tensor(arr) for arr in std]

        # 将这些张量堆叠为一个新的张量
        stacked_tensor = torch.stack(tensor_list)

        # 调整形状为[4, 1]
        final_tensor = stacked_tensor.unsqueeze(1)
        # 返回标准差的均值作为CLIP consensus指标
        return final_tensor
    


class InferenceDataset(Dataset):
    def __init__(self, datadir, inference_dir, test_scene, clip_preprocess, eval_resolution=256, img_suffix='.jpg', inpainted_suffix='_removed.png'):
        self.inference_dir = inference_dir
        self.datadir = datadir
        if not datadir.endswith('/'):
            datadir += '/'
        self.mask_filenames = sorted(list(glob.glob(os.path.join(self.datadir, '**', '*mask*.png'), recursive=True)))
        self.img_filenames = [fname.rsplit('_mask', 1)[0] + img_suffix for fname in self.mask_filenames]
        self.file_names = [os.path.join(inference_dir, os.path.splitext(fname[len(datadir):])[0] + inpainted_suffix)
                                for fname in self.img_filenames]

        
        self.clip_preprocess = clip_preprocess
        self.eval_resolution = eval_resolution
        self.test_scene = self.read_csv_to_dict(test_scene)
        self.ids = [file_name.rsplit('/', 1)[1].rsplit('_mask.png', 1)[0] for file_name in self.mask_filenames]

        self.collect_all_classes()

    def __len__(self):
        return len(self.ids)


    def read_csv_to_dict(self,file_path):
        data_dict = {}
        
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=',')
            header = next(reader)  # Skip header if there is one
            for row in reader:
                id = row[0].rsplit('.', 1)[0]
                LabelName = row[1]
                BoxXMin = float(row[2])
                BoxXMax = float(row[3])
                BoxYMin = float(row[4])
                BoxYMax = float(row[5])
                
                data_dict[id] = {
                    'LabelName': LabelName,
                    'BoxXMin': BoxXMin,
                    'BoxXMax': BoxXMax,
                    'BoxYMin': BoxYMin,
                    'BoxYMax': BoxYMax
                }
                
        return data_dict

    def scale_box(self, box, scale_ratio):
        return list(map(lambda x: int(x * scale_ratio), box))

    def get_cropped_boundary(self, object_bbox, image_size_orig):
        width, height = image_size_orig
        min_size = min(image_size_orig)
        object_bbox[0] -= (width - min_size) // 2
        object_bbox[1] -= (height - min_size) // 2
        object_bbox[2] -= (width - min_size) // 2
        object_bbox[3] -= (height - min_size) // 2
        object_bbox = np.clip(object_bbox, 0, min_size)
        return object_bbox

    def get_scaled_boundary(self, object_bbox, scale_ratio):
        object_bbox = np.array(self.scale_box(object_bbox, scale_ratio))
        return object_bbox

    def read_image(self, path):
        img = Image.open(path).resize((self.eval_resolution,self.eval_resolution), Image.Resampling.BILINEAR)
        return img

    def collect_all_classes(self):
        classes = set()
        for scene_id in self.ids:
            classes.add(self.test_scene[scene_id]["LabelName"])
        self.classes = list(classes)

    def add_padding(self, image):	
        padding_color = (0,0,0)
        width, height = image.size	
        if width > height:	
            padded_image = Image.new(image.mode, (width, width), padding_color)	
            padded_image.paste(image, (0, (width - height) // 2))	
        else:	
            padded_image = Image.new(image.mode, (height, height), padding_color)	
            padded_image.paste(image, ((height - width) // 2, 0))	
        return padded_image

    def __getitem__(self, idx):
        scene_id = self.ids[idx]
        object_bbox = (int(self.test_scene[scene_id]["BoxXMin"]*self.eval_resolution),
                       int(self.test_scene[scene_id]["BoxYMin"]*self.eval_resolution),
                       int(self.test_scene[scene_id]["BoxXMax"]*self.eval_resolution),
                       int(self.test_scene[scene_id]["BoxYMax"]*self.eval_resolution))
        object_name = self.test_scene[scene_id]["LabelName"]

        source_image = self.read_image(self.img_filenames[idx])

        inpainted_image = self.read_image(self.file_names[idx])

        
        #object_bbox = self.get_cropped_boundary(object_bbox, image_size_orig)
        
        #scale_ratio =  self.eval_resolution / min(image_size_orig)
        #object_bbox = np.array(self.scale_box(object_bbox, scale_ratio))
        return (
            self.clip_preprocess(self.add_padding(source_image.crop(object_bbox))),	
            self.clip_preprocess(self.add_padding(inpainted_image.crop(object_bbox))),
            object_name,
            scene_id,
        )
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datadir",
        type=str,
        default="outputs/gqa_inpaint_inference/",
        help="Directory of the original images and masks",
    )
    parser.add_argument(
        "--inference_dir",
        type=str,
        default="outputs/gqa_inpaint_inference/",
        help="Directory of the inference results",
    )
    parser.add_argument(
        "--test_scene",
        type=str,
        default="/hy-tmp/DATA/fetch_output.csv",
        help="path of the test scene",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/gqa_inpaint_eval/",
        help="Directory of evaluation outputs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Bath size of CLIP forward pass",
    )
    parser.add_argument(
        "--inpainted_suffix",
        type=str,
        default='_removed.png',
        help="inference_dir's suffix",
    )
    args = parser.parse_args()

    clip_metric = CLIPMetric(model_name="ViT-B/32")

    dataset = InferenceDataset(args.datadir, args.inference_dir, args.test_scene, clip_metric.preprocess ,eval_resolution=256, img_suffix='.png', inpainted_suffix=args.inpainted_suffix)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    inference_scores = {}
    scene_ids = []
    for idx, (source_img, inpainted_image, object_names, scene_id) in enumerate(tqdm(dataloader)):
        scene_ids.extend(list(scene_id))
        prompts = list(map(lambda x: f"a photo of a {x}", object_names))
        bk_prompts = ['background']*len(prompts)
        src_scores = clip_metric.score(source_img, prompts)
        #prd_scores  = clip_metric.score(inpainted_image, bk_prompts)
        prd_scores  = clip_metric.score(inpainted_image, prompts)
        for src_score, prd_seed1 , id in zip(src_scores, prd_scores, scene_id):
            inference_scores[id] = {
                "src_scores": src_score.item(),
                "prd_scores": prd_seed1.item(),
                "prd_clip_distance": src_score.item() - prd_seed1.item()
        }


    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    df_inference = pd.DataFrame.from_dict(inference_scores, orient='index', columns=['src_scores', 'prd_scores', 'prd_clip_distance']).set_index([scene_ids])
    df_inference.to_csv(f"{output_dir}/inference_scores.csv")

    column_means = df_inference.mean()
    print(column_means)
    with open(f"{output_dir}/clip_distance.txt", 'w') as f:
        f.write(column_means.to_string())
    print(f"output to {output_dir}clip_distance.txt ")

