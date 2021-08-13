import sys
import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from Transformer import CustomImage
from Visual_explanations import GradCam, GradCamplusplus, Guided_BackPropagation, Guided_GradCam

model = torchvision.models.resnet152(pretrained=True)
# model = torchvision.models.densenet161(pretrained=True)
model.eval()

Image_path = 'Data_samples\\150_sealion_and_145_penguin.jpg'
target_label_index = 150
original_img = Image.open(Image_path)
original_w, original_h = original_img.size[0], original_img.size[1]

Image_trans = CustomImage()
Input_img = Image_trans(Image_path)
Input_img = Input_img.unsqueeze(dim=0)

# ####################
# #### Recommend to run below three lines before executing Gradcam to find target layer's name and model's prediction results.
# # GC.get_names()
# # print(torch.argmax(GC.get_model_output(Input_img)))
# # sys.exit("Find the target layers and their name. For general GradCam and GradCam ++ results, target the last convolution layer in the model.")
# # ######################
#

target_layers = 'layer4.2.conv3'  # The last convolution layer of the resnet152 model.
# target_layers = 'features.norm5'  # The last convolution layer of the densenet161 model.

########## Demo 1. ##########
# # Run basic attribution methods GradCam, Gradcam++, Guided back propagation, Guided GradCam and Guided Gradcam++.

# 1. Gradcam
GC = GradCam(model)
GC_grads = GC.get_gradient(input_TensorImage=Input_img, target_layers=target_layers, target_label=target_label_index)
GC_vis = GC.visualize(GC_grads, original_img, view=True, save=False)
sys.exit()

# 2. Gradcam++
GCplpl = GradCamplusplus(model)
GCplpl_grads = GCplpl.get_gradient(input_TensorImage=Input_img, target_layers=target_layers, target_label=target_label_index)
GCplpl_vis = GCplpl.visualize(GCplpl_grads, original_img, view=True, save=False)
sys.exit()

# 3. Guided Back propagation
GBP = Guided_BackPropagation(model)
GBP_grad = GBP.get_gradient(input_TensorImage=Input_img, target_label=target_label_index)
GBP_grad = (GBP_grad - torch.min(GBP_grad))/torch.max(GBP_grad - torch.min(GBP_grad))
GBP_vis = GBP.visualize(GBP_grad, resize=[original_h, original_w], view=True, save=False)
sys.exit()

# 4. Guided Gradcam
GGC = Guided_GradCam(model)
GGC.visualize(GBP_grad=GBP_grad, GC_grads=GC_grads, resize=[original_h, original_w], view=True,  save=False)
sys.exit()

# 5. Guided Gradcam++
GGCplpl = Guided_GradCam(model)
GGCplpl.visualize(GBP_grad=GBP_grad, GC_grads=GCplpl_grads, view=True, resize=[original_h, original_w], save=False)
sys.exit()


######### Demo 2. ##########
# Generating Gradcam images from 4 layers in the model.

# Layers of the resnet152 model.
layer1 = 'layer1.0.conv1'
layer2 = 'layer2.2.conv3'
layer3 = 'layer3.2.conv3'
layer4 = 'layer4.2.conv3'

# # Layers of the densenet161 model.
# layer1 = 'features.transition1.norm'
# layer2 = 'features.transition2.norm'
# layer3 = 'features.transition3.norm'
# layer4 = 'features.norm5'

target_layers = [layer1, layer2, layer3, layer4]
file_names = ['layer1.png', 'layer2.png', 'layer3.png', 'layer4.png']

GC = GradCam(model)
GC_grads = GC.get_gradient(input_TensorImage=Input_img, target_layers=target_layers, target_label=target_label_index)
GC_vis = GC.visualize(GC_grads, original_img, view=True, save=False)
sys.exit()


########## Demo 3. ##########
# Generating counterfactual explanations images of GradCam.
# And additionally, by applying counterfactual explanations algorithm on the Gradcam++, generating counterfactual explanations images of GradCam++.
# Results are unstable and heavily dependent on the model and data. ###

target_layers = 'layer4.2.conv3'  # The last convolution layer of the resnet152 model.
# target_layers = 'features.norm5'  # The last convolution layer of the densenet161 model.
# Gradcam_counterfactual
GC = GradCam(model)
GC_grads = GC.get_gradient(input_TensorImage=Input_img, target_layers=target_layers, counter=True, target_label=target_label_index)
GC_vis = GC.visualize(GC_grads, original_img, view=True, save=False)
