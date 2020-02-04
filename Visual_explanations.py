import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from collections.abc import Iterable

class _Base:
    def __init__(self, model):
        super(_Base, self).__init__()
        self.model = model
        self.model.eval()  # model have to get .eval() for evaluation.

    def normalization(self, x):
        x -= x.min()
        if x.max() <=0.:
            x /= 1.  # to avoid Nan
        else:
            x /= x.max()
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

    def get_gradient(self, input_TensorImage, target_layers, target_label=None, input_hook=False):
        """
        This function is base for Gradcam.get_gradient and Gradcamplusplus.get_gradient.

        :param input_TensorImage (tensor): Input Tensor image with [1, c, h, w].
        :param target_layers (str, list): Names of target layers. Can be set to string for a layer, to list for multiple layers, or to "All" for all layers in the model.
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
        self.target_layers = target_layers

        if not input_TensorImage.dim() == 4: raise NotImplementedError("input_TensorImage must be 4-dimension.")
        if not input_TensorImage.size()[0] == 1: raise NotImplementedError("batch size of input_TensorImage must be 1.")

        if not target_layers == 'All':
            if isinstance(target_layers, str) or not isinstance(target_layers, Iterable):
                self.target_layers = [self.target_layers]
                for target_layer in self.target_layers:
                    if not isinstance(target_layer, str):
                        raise NotImplementedError(
                            " 'Target layers' or 'contents in target layers list' are must string format.")

        for name, module in self.model.named_modules():
            if target_layers == 'All':
                if isinstance(module, nn.Conv2d):
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

    def visualize(self, gradients, original_input_image=None, view=True, save_locations=None):
        """
        Visualize gradient maps

        :param original_input_image (PIL): Original input image.
        :param view (bool): If True, will show a result. Default: True
        :param save_loacations (string, list, optional): Path of save locations and file names. Default: None
        :return (list): A list concluding gradient maps of target layers.
        """
        if save_locations:
            if not isinstance(save_locations, list): save_locations = [save_locations]
            if not len(gradients) == len(save_locations):
                raise NotImplementedError("Numbers of target_layer and save_locations is different.")
            for save_location in save_locations:
                if not isinstance(save_location, str): raise NotImplementedError("Locations in save_locations are must string.")

        if isinstance(original_input_image, PIL.Image.Image):
            width, height = original_input_image.size[0], original_input_image.size[1]
        else:
            raise NotImplementedError('original_input_image is a must PIL.image.image with [H, W, C] shape.')

        gradmaps = []
        for idx, gradient in enumerate(gradients):
            interpolated_gradmap = F.interpolate(gradient, (height, width), mode='bilinear', align_corners=False)
            interpolated_gradmap = interpolated_gradmap[0][0]
            gradmaps.append(interpolated_gradmap)

            if view:
                plt.imshow(original_input_image)
                plt.imshow(interpolated_gradmap, cmap='jet', alpha=0.5)
                plt.axis('off')
                plt.tight_layout()
                if save_locations: fig = plt.gcf()
                plt.show()

            if save_locations:
                fig.savefig(save_locations[idx])

        return gradmaps


class GradCam(_Base):
    def __init__(self, model):
        super(GradCam, self).__init__(model)

    def get_gradient(self, input_TensorImage, target_layers, target_label=None, counter=False, input_hook=False):
        """
        Get backward-propagation gradient.

        :param counter (bool): If True, will get negative gradients only for conterfactual explanations. Default: True
        :return (list): A list including gradients of Gradcam for target layers
        """
        super(GradCam, self).get_gradient(input_TensorImage, target_layers, target_label=target_label, input_hook=input_hook)

        def process():
            grads = self.backward_out[name]
            if counter:
                grads = torch.clamp(grads, max=0.)
                grads *= -1.
            weight = torch._adaptive_avg_pool2d(grads, 1)
            gradient = self.forward_out[name] * weight
            gradient = gradient.sum(dim=1, keepdim=True)
            gradient = F.relu(gradient)
            gradient = self.normalization(gradient)
            self.gradients.append(gradient)

        if not target_layers == 'All':
            for name in self.target_layers:
                process()
        else:
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Conv2d):
                    process()

        return self.gradients


class GradCamplusplus(_Base):
    def __init__(self, model):
        super(GradCamplusplus, self).__init__(model)

    def get_gradient(self, input_TensorImage, target_layers, target_label=None, counter=False, input_hook=False):
        """
        Get backward-propagation gradient.

        :param counter (bool): If True, will get negative gradients only for conterfactual explanations. Default: True
        :return (list): A list including gradients of Gradcam++ for target layers
        """
        super(GradCamplusplus, self).get_gradient(input_TensorImage, target_layers, target_label=target_label, input_hook=input_hook)

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
            gradient = self.normalization(gradient)
            self.gradients.append(gradient)

        if not target_layers == 'All':
            for name in self.target_layers:
                process()
        else:
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Conv2d):
                    process()

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
        self.guided_gradient = self.normalization(self.guided_gradient)

        return self.guided_gradient

    def visualize(self, gradient, view=False, resize=None, save_location=None):
        """
        :param view (bool): If True, will show a result. Default: True
        :param resize (list, optional): Determine size of resizing image with list [height, width]. Default: None
        :param save_loacations (string, list, optional): Path of save locations and file names. Default: None
        :return (tensor): Guided-BackPropagation gradient of the input image.
        """
        if save_location:
            if not isinstance(save_location, list): save_location = [save_location]
            for location in save_location:
                if not isinstance(location, str): raise NotImplementedError("save_locations are must string.")

        if resize:
            gradient = F.interpolate(gradient, (resize[0], resize[1]), mode='bilinear', align_corners=False)
        gradient = torch.squeeze(gradient, dim=0)
        gradient = gradient.permute(1, 2, 0)

        if view:
            plt.imshow(gradient, cmap='jet')
            plt.axis('off')
            plt.tight_layout()
            if save_location: fig = plt.gcf()
            plt.show()

        if save_location:
            fig.savefig(save_location[0])

        return gradient


class Guided_GradCam(Guided_BackPropagation):
    def __init__(self):
        pass

    def visualize(self, GBP_grad = None, GC_grads = None, view=False, resize=None, save_locations=None):
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

        if save_locations:
            if not isinstance(save_locations, list): save_locations = [save_locations]
            if not len(GC_grads) == len(save_locations):
                raise NotImplementedError("Numbers of target_layers of Gradcam and save_locations is different.")
            for save_location in save_locations:
                if not isinstance(save_location, str): raise NotImplementedError("save_locations are must string.")

        for i in range(len(GC_grads)):
            save_location = save_locations[i] if save_locations else None

            GC_grad = GC_grads[i]
            GC_grad = F.interpolate(GC_grad, (GBP_grad.size()[-2], GBP_grad.size()[-1]), mode='bilinear', align_corners=False)
            gradient = GBP_grad * GC_grad
            gradient = self.normalization(gradient)

            gradient -= torch.mean(gradient)
            gradient /= torch.std(gradient)+1e-5
            gradient *= 0.1
            gradient += 0.5
            gradient = torch.clamp(gradient, min=0, max=1)
            super(Guided_GradCam, self).visualize(gradient, view=view, save_location=save_location, resize=resize)
