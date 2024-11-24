classes = ['Mandarine', 'Apple Braeburn', 'Apple Red', 'Papaya', 'Nectarine', 'Pomegranate',
            'Apple Golden', 'Grape White', 'Cocos', 'Tomato', 'Plum', 'Cucumber Ripe', 'Eggplant',
            'Cucumber', 'Zucchini', 'Apple Granny Smith', 'Pineapple', 'Peach', 'Limes', 'Peach',
            'Fig', 'Banana', 'Lemon', 'Cherry', 'Cherry','Kiwi', 'Watermelon', 'Mango Red', 
            'Pear', 'Strawberry', 'Passion Fruit', 'Apple Red','Apple Golden', 'Blueberry',
            'Mango', 'Pepper Red', 'Orange', 'Raspberry', 'Avocado', 'Avocado ripe']
classes.sort()

import torch
import torch.nn as nn
import os
from PIL import Image
import torchvision.transforms as tt
import torch.nn.functional as F

class FruitsVegCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.AdaptiveAvgPool2d((1, 1)), #Global Average Pooling
            nn.Flatten(),
            nn.Dropout(0.5),  #Increased dropout
            nn.Linear(128, len(classes)) #Much smaller linear layer
        )

    def forward(self, xb):
        return self.network(xb)


model_path = "fruitveg_final.pth"
print(f"Loading model from: {os.path.abspath(model_path)}")
model = FruitsVegCNN()    
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

transforms = tt.Compose([
    tt.Resize(100),
    tt.ToTensor()
])

def predict_img(img):
    if img.mode == 'RGBA':  # Check if the image has an alpha channel
        # Create a white background in RGBA mode
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img).convert("RGB") #Replace transparent bg with white
    img = transforms(img)
    img = img.unsqueeze(0)
    
    predicted_results = model(img)
    probabilities = F.softmax(predicted_results, dim=1)
    probability, index = torch.topk(probabilities, 1, dim=1)
    print("Predicted: ", classes[index], " with confidence: ", probability[0].item())
    return ("Predicted:     "+ str(classes[index])+ " with confidence:  "+ str(probability[0].item()))

#Setting up streamlit
import streamlit as st
st.title("Fruits & Vegetable prediction system")
df = st.file_uploader(label= "Upload your file")
if(df):
    img = Image.open(df)
    st.image(img, width=300)
    st.write(predict_img(img))


