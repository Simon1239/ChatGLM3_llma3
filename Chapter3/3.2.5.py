#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# step1:模型参数的载入
import torch, torchvision

# 加载与训练中使用的结构相同的模型
def load_model(checkpoint_path, num_classes=2, device='cuda'):
    model = torchvision.models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

# step2：预测函数的编写
from torchvision import transforms

def preprocess_image(image):
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = image_transform(image).unsqueeze(0)
    return image_transform(image)

model = load_model("./checkpoint/latest_checkpoint.pth")
def predict(image):
    classes = {"0":"cat", "1":"dog"}
    image = preprocess_image(image)
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1).squeeze()
    # 将类标签映射到概率值
    class_probabilities = {classes[str(i)]: float(prob) for i, prob in enumerate(probabilities)}
    return class_probabilities

# step3:可视化图像的编写
import gradio as gr
iface = gr.Interface(
    fn=predict,
    inputs = gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="Cat VS Dog Classifier",
)

if __name__ == "__main__":
    iface.launch()
