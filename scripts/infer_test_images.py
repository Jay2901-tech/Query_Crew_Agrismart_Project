import torch, timm, json
from pathlib import Path
from PIL import Image
from torchvision import transforms

CHECKPOINT = "models/experiment_1/best_model.pth"
IMAGES = ["test1.jpg", "test2.jpg", "test3.jpg", "test4.jpg"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load(CHECKPOINT, map_location=device)
classes = ckpt['classes']

model = timm.create_model('tf_efficientnetv2_s', pretrained=False, num_classes=len(classes))
model.load_state_dict(ckpt['model_state_dict'])
model.eval().to(device)

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

for img_path in IMAGES:
    img = Image.open(img_path).convert('RGB')
    inp = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(inp), dim=1)[0]
    top5 = probs.topk(5)
    print(f"\n{'='*40}")
    print(f"Image: {img_path}")
    print(f"  #1  {classes[top5.indices[0]]:<45} {top5.values[0]*100:.2f}%")
    for i in range(1, 5):
        print(f"  #{i+1}  {classes[top5.indices[i]]:<45} {top5.values[i]*100:.2f}%")
