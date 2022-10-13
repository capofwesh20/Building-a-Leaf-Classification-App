import datasets
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import gradio as gr
import torch 

feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
dataset = datasets.load_dataset("beans")

extractor = AutoFeatureExtractor.from_pretrained("capofwesh20/bean_leaf_classifier")
model = AutoModelForImageClassification.from_pretrained("capofwesh20/bean_leaf_classifier")

labels = dataset['train'].features['labels'].names

def classify(im):
  features = feature_extractor(im, return_tensors='pt')
  logits = model(features["pixel_values"])[-1]
  probability = torch.nn.functional.softmax(logits, dim=-1)
  probs = probability[0].detach().numpy()
  confidences = {label: float(probs[i]) for i, label in enumerate(labels)} 
  return confidences



sample_images=[['https://s3.amazonaws.com/moonup/production/uploads/1663933284359-611f9702593efbee33a4f7c9.png'],
['https://s3.amazonaws.com/moonup/production/uploads/1663933284374-611f9702593efbee33a4f7c9.png'],
['https://s3.amazonaws.com/moonup/production/uploads/1663933284412-611f9702593efbee33a4f7c9.png']]

title = 'Bean Leaf Classifier'
description = 'This model is trained for beans leaf classification but might give a false result on other leaves'
interface = gr.Interface(classify, gr.Image(shape=(200, 200)), 'label',
                         title = title,
                         description = description,
                         examples=sample_images)

interface.launch()
