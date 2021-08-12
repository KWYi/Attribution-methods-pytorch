import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from collections.abc import Iterable
import pickle

def Nancheck(x):
    if (x != x).any():
        print('NaN')
    else:
        print("No NaN")


"""
Usage example:

from PIL import Image
from Visual_explanations import Gradcam , Gradcamplusplus, Guided_BackPropagation, Guided_gradcam

Original_image = Image.open('.....') # [h,w,c]
Tensor_image = transformed image form Original_image # [b,c,h,w]

example_Model = model what you trained

########## Finding target layer's name in the model ##########
finding = Gradcam(example_model)  # or finding = Gradcamplusplus(model) or finding = Guided_BackPropagation(example_model)
finding.get_names()


########## Gradcam // Gradcam++ ##########

GC = Gradcam(example_model)  # or GC = Gradcamplusplus(model)
target_layer = ['layer3.2.conv3', 'layer4.2.conv3']
GC_grad = GC.get_gradient(input_TensorImage=Tensor_image, target_layers=target_layer, target_label=1, input_hook=False)
GC.visualize(GC_grad, original_input_image=original_img, view=True)


########## Guided Back propagation ##########

GBP = Guided_BackPropagation(example_model)
output = GBP.get_model_output(Input)
GBP_grad = GBP.get_gradient(input_TensorImage=Input, target_label=14)
GBP.visualize(GBP_grad, resize=[original_h, original_w], view=True)


########## Guided Back propagation ##########

GGC = Guided_gradcam()
GGC.visualize(GBP_grad=GBP_grad, GC_grads=GC_grads, view=True, resize=[original_h, original_w])


"""

class _Base:
    def __init__(self, model):
        super(_Base, self).__init__()
        self.model = model
        self.model.eval()  # model have to get .eval() for evaluation.

    def normalization(self, x):
        x = x - x.min()
        if x.max() <=0.:
            pass  # to avoid Nan
        else:
            x = x / x.max()
        return x

    def forward_hook(self, name, input_hook=False):
        def save_forward_hook(module, input, output):
            if input_hook:
                self.forward_out[name] = input[0].detach()
            else:
                self.forward_out[name] = output.detach()
        return save_forward_hook

    def backward_hook(self, name, input_hook=False):
        def save_backward_hook(module, grad_input, grad_output):
            if input_hook:
                self.backward_out[name] = grad_input[0].detach()
            else:
                self.backward_out[name] = grad_output[0].detach()
        return save_backward_hook

    def get_model_output(self, input_TensorImage):
        """ function to get result of the model. """
        self.model.zero_grad()
        result = self.model(input_TensorImage)

        return result

    def get_names(self):
        """ function to get names of layers in the model. """
        for name, module in self.model.named_modules():
            print(name, '//', module)

    def get_gradient(self, input_TensorImage, target_layers, target_layer_types, target_label, input_hook):
        """
        This function is base for Gradcam.get_gradient and Gradcamplusplus.get_gradient.

        :param input_TensorImage (tensor): Input Tensor image with [1, c, h, w].
        :param target_layers (str, list): Names of target layers. Can be set to string for a layer, to list for multiple layers, or to "All" for all layers in the model.
        :param target_layer_types (str, type, list, tuple): Define target layer's type when target_layers = 'All'
        :param target_label (int, tensor): Target label. If None, will determine index of highest label of the model's output as the target_label.
                                            Can be set to int as index of output, or to a Tensor that has same shape with output of the model. Default: None
        :param input_hook (bool): If True, will get input features and gradients of target layers instead of output. Default: False
        """
        if not isinstance(input_TensorImage, torch.Tensor):
            raise NotImplementedError('input_TensorImage is a must torch.Tensor format with [..., C, H, W]')
        self.model.zero_grad()
        self.forward_out = {}
        self.backward_out = {}
        self.handlers = []
        self.gradients = []
        self.gradients_min_max = []
        self.target_layers = target_layers
        self.target_layer_types = target_layer_types

        if not input_TensorImage.dim() == 4: raise NotImplementedError("input_TensorImage must be 4-dimension.")
        if not input_TensorImage.size()[0] == 1: raise NotImplementedError("batch size of input_TensorImage must be 1.")

        if not target_layers == 'All':
            if isinstance(target_layers, str) or not isinstance(target_layers, Iterable):
                self.target_layers = [self.target_layers]
                for target_layer in self.target_layers:
                    if not isinstance(target_layer, str):
                        raise NotImplementedError(
                            " 'Target layers' or 'contents in target layers list' are must string format.")
        else:
            if self.target_layer_types == 'All' or isinstance(self.target_layer_types, type) or isinstance(self.target_layer_types, tuple):
                pass
            elif isinstance(self.target_layer_types, list):
                self.target_layer_types = tuple(self.target_layer_types)
            else:
                raise NotImplementedError("'target_layer_types' must be 'All', type, list or tuple")

        for name, module in self.model.named_modules():
            if self.target_layers == 'All':
                if self.target_layer_types == 'All':
                    self.handlers.append(module.register_forward_hook(self.forward_hook(name, input_hook)))
                    self.handlers.append(module.register_backward_hook(self.backward_hook(name, input_hook)))

                elif isinstance(module, self.target_layer_types):
                    self.handlers.append(module.register_forward_hook(self.forward_hook(name, input_hook)))
                    self.handlers.append(module.register_backward_hook(self.backward_hook(name, input_hook)))
            else:
                if name in self.target_layers:
                    self.handlers.append(module.register_forward_hook(self.forward_hook(name, input_hook)))
                    self.handlers.append(module.register_backward_hook(self.backward_hook(name, input_hook)))

        output = self.model(input_TensorImage)

        if target_label is None:
            target_tensor = torch.zeros_like(output)
            target_tensor[0][int(torch.argmax(output))] = 1.
        else:
            if isinstance(target_label, int):
                target_tensor = torch.zeros_like(output)
                target_tensor[0][target_label] = 1.
            elif isinstance(target_label, torch.Tensor):
                if not target_label.dim() == output.dim():
                    raise NotImplementedError('Dimension of output and target label are different')
                target_tensor = target_label
        output.backward(target_tensor)

        self.model.zero_grad()
        for handle in self.handlers:
            handle.remove()

    def visualize(self, gradients, original_input_image=None, view=True, size=[1024, 1024], save=False, save_locations=None, save_names=None, width=None, height=None):
        """
        Visualize gradient maps

        :param original_input_image (PIL): Original input image.
        :param view (bool): If True, will show a result. Default: True
        :param size (list): Define size of window of image viewer. Default: [6.4, 4.8] (Default option of matplotlib)
        :param save (bool): If True, will save image files
        :param save_loacations (string, list, optional): Path of save locations and file names.
        :param save_names (string, list, optional): Path of save locations and file names.
        If save=True and save_location=None, files are saved using basic name. Default: None
        :return (list): A list concluding gradient maps of target layers.
        """
        if save_locations:
            if not isinstance(save_locations, list): save_locations = [save_locations]
            if len(save_locations) != len(gradients) and len(save_locations) != 1:
                raise NotImplementedError("Numbers of target_layer and save_locations is different.")
            for save_location in save_locations:
                if not isinstance(save_location, str): raise NotImplementedError("Locations in save_locations are must string.")

        if save_names:
            if not isinstance(save_names, list): save_names = [save_names]
            if not len(gradients) == len(save_names):
                raise NotImplementedError("Numbers of target_layer and save_locations is different.")
            for save_name in save_names:
                if not isinstance(save_name, str): raise NotImplementedError("Locations in save_locations are must string.")

        if original_input_image:
            if isinstance(original_input_image, PIL.Image.Image):
                width, height = original_input_image.size[0], original_input_image.size[1]
            else:
                raise NotImplementedError('Original Input_image is must PIL.image.image with [H, W, C] shape.')

        gradmaps = []
        for idx, gradient in enumerate(gradients):
            interpolated_gradmap = F.interpolate(gradient, (height, width), mode='bilinear', align_corners=False)
            interpolated_gradmap = interpolated_gradmap[0][0]
            interpolated_gradmap = interpolated_gradmap.cpu()
            ###############################
            # if not torch.max(interpolated_gradmap) == 1. or not torch.min(interpolated_gradmap) == 0.:
            #     interpolated_gradmap = self.normalization(interpolated_gradmap)
            ###############################
            gradmaps.append(interpolated_gradmap)
            plt.figure(figsize=(size[0], size[1]), dpi=1)

            if original_input_image:
                plt.imshow(original_input_image)
                plt.imshow(interpolated_gradmap, cmap='jet', alpha=0.5)
            else:
                plt.imshow(interpolated_gradmap, cmap='gray')
            plt.axis('off')
            plt.tight_layout()
            if save: fig = plt.gcf()
            if view: plt.show()

            if save:
                if save_locations:
                    if save_names:
                        if len(save_locations) == 1:
                            fig.savefig(save_locations[0] + '\\' + save_names[idx] + '.eps', bbox_inches='tight',
                                        pad_inches=0)
                            fig.savefig(save_locations[0] + '\\' + save_names[idx] + '.png', bbox_inches='tight',
                                        pad_inches=0)
                            with open(save_locations[0] + '\\' + save_names[idx] + '.pickle', 'wb') as f:
                                pickle.dump(gradient, f, pickle.HIGHEST_PROTOCOL)
                        else:
                            fig.savefig(save_locations[idx] + '\\' + save_names[idx] + '.eps', bbox_inches='tight',
                                        pad_inches=0)
                            fig.savefig(save_locations[idx] + '\\' + save_names[idx] + '.png', bbox_inches='tight',
                                        pad_inches=0)
                            with open(save_locations[idx] + '\\' + save_names[idx] + '.pickle', 'wb') as f:
                                pickle.dump(gradient, f, pickle.HIGHEST_PROTOCOL)
                    else:
                        if len(save_locations) == 1:
                            fig.savefig(save_locations[0] + '\\' + self.basic_names[idx] + '.eps', bbox_inches='tight',
                                        pad_inches=0)
                            fig.savefig(save_locations[0] + '\\' + self.basic_names[idx] + '.png', bbox_inches='tight',
                                        pad_inches=0)
                            with open(save_locations[0] + '\\' + self.basic_names[idx] + '.pickle', 'wb') as f:
                                pickle.dump(gradient, f, pickle.HIGHEST_PROTOCOL)
                        else:
                            fig.savefig(save_locations[idx] + '\\' + self.basic_names[idx] + '.eps',
                                        bbox_inches='tight', pad_inches=0)
                            fig.savefig(save_locations[idx] + '\\' + self.basic_names[idx] + '.png',
                                        bbox_inches='tight', pad_inches=0)
                            with open(save_locations[idx] + '\\' + self.basic_names[idx] + '.pickle', 'wb') as f:
                                pickle.dump(gradient, f, pickle.HIGHEST_PROTOCOL)
                else:
                    fig.savefig(self.basic_names[idx] + '.eps', bbox_inches='tight', pad_inches=0)
                    fig.savefig(self.basic_names[idx] + '.png', bbox_inches='tight', pad_inches=0)
                    with open(self.basic_names[idx] + '.pickle', 'wb') as f:
                        pickle.dump(gradient, f, pickle.HIGHEST_PROTOCOL)
            plt.close()
        return gradmaps


class GradCam(_Base):
    def __init__(self, model):
        super(GradCam, self).__init__(model)
        self.cam = []
        self.N_gradients = []

    def get_gradient(self, input_TensorImage, target_layers, target_layer_types='All', target_label=None, input_hook=False, counter=False):
        """
        Get backward-propagation gradient.

        :param counter (bool): If True, will get negative gradients only for conterfactual explanations. Default: True
        :return (list): A list including gradients of Gradcam for target layers
        """
        super(GradCam, self).get_gradient(input_TensorImage, target_layers, target_layer_types=target_layer_types, target_label=target_label, input_hook=input_hook)

        def process():
            grads = self.backward_out[name]
            if counter:
                grads = torch.clamp(grads, max=0.)
                grads *= -1.
            weight = torch._adaptive_avg_pool2d(grads, 1)
            gradient = self.forward_out[name] * weight
            gradient = gradient.sum(dim=1, keepdim=True)
            self.cam.append(gradient)
            self.N_gradients.append(F.relu(gradient * -1))
            gradient = F.relu(gradient)
            self.gradients_min_max.append([torch.min(gradient), torch.max(gradient)])
            # gradient = self.normalization(gradient)
            self.gradients.append(gradient)

        self.basic_names = []

        if not target_layers == 'All':
            for name in self.target_layers:
                process()
                self.basic_names += [name]
        else:
            for name, module in self.model.named_modules():
                if self.target_layer_types == 'All':
                    process()
                elif isinstance(module, self.target_layer_types):
                    process()
                self.basic_names += [name]

        return self.gradients


class GradCamplusplus(_Base):
    def __init__(self, model):
        super(GradCamplusplus, self).__init__(model)

    def get_gradient(self, input_TensorImage, target_layers, target_layer_types='All', target_label=None, input_hook=False, counter=False):
        """
        Get backward-propagation gradient.

        :param counter (bool): If True, will get negative gradients only for conterfactual explanations. Default: True
        :return (list): A list including gradients of Gradcam++ for target layers
        """
        super(GradCamplusplus, self).get_gradient(input_TensorImage, target_layers, target_layer_types=target_layer_types, target_label=target_label, input_hook=input_hook)

        def process():
            features = self.forward_out[name]
            grads = self.backward_out[name]
            if counter:
                grads *= -1.
            relu_grads = F.relu(grads)
            alpha_numer = grads.pow(2)
            alpha_denom = 2. * grads.pow(2) + grads.pow(3) * features.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)
            alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
            alpha = alpha_numer / alpha_denom
            weight = (alpha * relu_grads).sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)
            gradient = features * weight
            gradient = gradient.sum(dim=1, keepdim=True)
            gradient = F.relu(gradient)
            # gradient = self.normalization(gradient)
            self.gradients.append(gradient)

        self.basic_names = []
        if not target_layers == 'All':
            for name in self.target_layers:
                process()
                self.basic_names += [name]
        else:
            for name, module in self.model.named_modules():
                if self.target_layer_types == 'All':
                    process()
                elif isinstance(module, self.target_layer_types):
                    process()
                self.basic_names += [name]

        return self.gradients


class Guided_BackPropagation(_Base):
    def __init__(self, model):
        super(Guided_BackPropagation, self).__init__(model)

    def relu_backward_hook(self, module, grad_input, grad_output):
        return (torch.clamp(grad_input[0], min=0.), )

    def get_gradient(self, input_TensorImage, target_label=None):
        """
        :param input_TensorImage (tensor): Input Tensor image with [1, c, h, w].
        :param target_label (int, tensor): Target label. If None, will determine index of highest label of the model's output as the target_label.
                                            Can be set to int as index of output, or to a Tensor that has same shape with output of the model. Default: None
        :return (tensor): Guided-BackPropagation gradients of the input image.
        """
        self.model.zero_grad()
        self.guided_gradient = None
        self.handlers = []
        self.gradients = []
        self.gradients_min_max = []
        self.input_TensorImage = input_TensorImage.requires_grad_()

        for name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                self.handlers.append(module.register_backward_hook(self.relu_backward_hook))

        output = self.model(self.input_TensorImage)

        if target_label is None:
            target_tensor = torch.zeros_like(output)
            target_tensor[0][torch.argmax(output)] = 1.
        else:
            if isinstance(target_label, int):
                target_tensor = torch.zeros_like(output)
                target_tensor[0][target_label] = 1.
            elif isinstance(target_label, torch.Tensor):
                if not target_label.dim() == output.dim():
                    raise NotImplementedError('Dimension of output and target label are different')
                target_tensor = target_label

        output.backward(target_tensor)

        for handle in self.handlers:
            handle.remove()

        self.guided_gradient = self.input_TensorImage.grad.clone()
        self.input_TensorImage.grad.zero_()
        self.guided_gradient.detach()
        self.guided_gradient_abs = torch.abs(self.guided_gradient)
        self.guided_gradient_relu = torch.relu(self.guided_gradient)
        self.guided_gradient_before_norm = self.guided_gradient.clone()

        self.gradients_min_max.append([torch.min(self.guided_gradient), torch.max(self.guided_gradient)])

        # self.guided_gradient = self.normalization(self.guided_gradient)
        # self.guided_gradient_abs = self.normalization((self.guided_gradient_abs))
        # self.guided_gradient_relu = self.normalization(self.guided_gradient_relu)

        return self.guided_gradient

    def visualize(self, gradient, view=False, size=[1024, 1024], resize=None, save=False, save_locations=None, save_names=None):
        """
        :param view (bool): If True, will show a result. Default: True
        :param resize (list, optional): Determine size of resizing image with list [height, width]. Default: None
        :param save_loacations (string, list, optional): Path of save locations and file names. Default: None
        :return (tensor): Guided-BackPropagation gradient of the input image.
        """
        if save_locations:
            if not isinstance(save_locations, list): save_locations = [save_locations]
            for save_location in save_locations:
                if not isinstance(save_location, str): raise NotImplementedError("Locations in save_locations are must string.")

        if save_names:
            if not isinstance(save_names, list): save_names = [save_names]
            for save_name in save_names:
                if not isinstance(save_name, str): raise NotImplementedError("Locations in save_locations are must string.")

        if resize:
            gradient = F.interpolate(gradient, (resize[0], resize[1]), mode='bilinear', align_corners=False)
        gradient = torch.squeeze(gradient, dim=0)
        gradient = gradient.permute(1, 2, 0)
        gradient = gradient.cpu()
        gradient = gradient.squeeze()

        plt.figure(figsize=(size[0], size[1]), dpi=1)
        plt.imshow(gradient, cmap='gray')
        plt.axis('off')
        plt.tight_layout()

        if save: fig = plt.gcf()
        if view: plt.show()
        self.basic_names = ['Guided_back_propagation']
        if save:
            if save_locations:
                if save_names:
                    if len(save_locations) == 1:
                        fig.savefig(save_locations[0] + '\\' + save_names[0] + '.eps', bbox_inches='tight',
                                    pad_inches=0)
                        fig.savefig(save_locations[0] + '\\' + save_names[0] + '.png', bbox_inches='tight',
                                    pad_inches=0)
                        with open(save_locations[0] + '\\' + save_names[0] + '.pickle', 'wb') as f:
                            pickle.dump(gradient, f, pickle.HIGHEST_PROTOCOL)
                    else:
                        fig.savefig(save_locations[0] + '\\' + save_names[0] + '.eps', bbox_inches='tight',
                                    pad_inches=0)
                        fig.savefig(save_locations[0] + '\\' + save_names[0] + '.png', bbox_inches='tight',
                                    pad_inches=0)
                        with open(save_locations[0] + '\\' + save_names[0] + '.pickle', 'wb') as f:
                            pickle.dump(gradient, f, pickle.HIGHEST_PROTOCOL)
                else:
                    if len(save_locations) == 1:
                        fig.savefig(save_locations[0] + '\\' + self.basic_names[0] + '.eps', bbox_inches='tight',
                                    pad_inches=0)
                        fig.savefig(save_locations[0] + '\\' + self.basic_names[0] + '.png', bbox_inches='tight',
                                    pad_inches=0)
                        with open(save_locations[0] + '\\' + self.basic_names[0] + '.pickle', 'wb') as f:
                            pickle.dump(gradient, f, pickle.HIGHEST_PROTOCOL)

                    else:
                        fig.savefig(save_locations[0] + '\\' + self.basic_names[0] + '.eps', bbox_inches='tight',
                                    pad_inches=0)
                        fig.savefig(save_locations[0] + '\\' + self.basic_names[0] + '.png', bbox_inches='tight',
                                    pad_inches=0)
                        with open(save_locations[0] + '\\' + self.basic_names[0] + '.pickle', 'wb') as f:
                            pickle.dump(gradient, f, pickle.HIGHEST_PROTOCOL)
            else:
                fig.savefig(self.basic_names[0] + '.eps', bbox_inches='tight', pad_inches=0)
                fig.savefig(self.basic_names[0] + '.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        return gradient


class Guided_GradCam(Guided_BackPropagation):
    def __init__(self, model):
        super(Guided_GradCam, self).__init__(model)
        pass

    def visualize(self, GBP_grad = None, GC_grads = None, view=False, size=[1024, 1024], resize=None, save=False, save_locations=None, save_names=None):
        """
        Visualize and/or save Guided gradcam or Guided gradcam++.
        To use this function, excute Guided_BackPropagation and Gradcam or Gradcamplusplus first, and input them in this function.

        :param GBP_grad: Gradients from GuidedBackPropagation.
        :param GC_grads:Gradients from GradCam or GradCam++.
        :param view (bool): If True, will show a result. Default: True
        :param resize (list, optional): Determine size of resizing image with list [height, width]. Default: None
        :param save_loacations (string, list, optional): Path of save locations and file names. Default: False
        """
        if GBP_grad is None or GC_grads is None:
            raise NotImplementedError(
                "Pleas execute Guided_BackPropagation and Gradcam or Gradcamplusplus first and input them in GBP_grad= and GC_grad.")
        if not isinstance(GC_grads, list):
            GC_grads = [GC_grads]

        if save_locations:
            if not isinstance(save_locations, list): save_locations = [save_locations]
            if len(save_locations) != len(GC_grads) and len(save_locations) != 1:
                raise NotImplementedError("Numbers of target_layers of Gradcam and save_locations is different.")
            for save_location in save_locations:
                if not isinstance(save_location, str): raise NotImplementedError("save_locations are must string.")

        if save_names:
            if not isinstance(save_names, list): save_names = [save_names]
            if not len(GC_grads) == len(save_names):
                raise NotImplementedError("Numbers of target_layer and save_locations is different.")
            for save_name in save_names:
                if not isinstance(save_name, str): raise NotImplementedError("Locations in save_locations are must string.")
        else:
            save_names = ['Combined']

        for GC_grad in GC_grads:
            GC_grad = F.interpolate(GC_grad, (GBP_grad.size()[-2], GBP_grad.size()[-1]), mode='bilinear', align_corners=False)
            gradient = GBP_grad * GC_grad
            # gradient = self.normalization(gradient)

            gradient -= torch.mean(gradient)
            gradient /= torch.std(gradient)+1e-5
            gradient *= 0.1
            gradient += 0.5
            gradient = torch.clamp(gradient, min=0, max=1)
            GGC_result = super(Guided_GradCam, self).visualize(gradient, view=view, size=size, resize=resize, save=save,
                                                             save_locations=save_locations, save_names=save_names)
            return GGC_result
