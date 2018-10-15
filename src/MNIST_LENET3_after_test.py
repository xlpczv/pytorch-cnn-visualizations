
# coding: utf-8

# In[1]:


#### History board
# After test code
# Vanilla and Guided Backpropagation
# Eigen and original dimension of neurons


# In[2]:


from __future__ import print_function
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ReLU
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import os
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from PIL import Image as PILImage
from IPython.display import Image
from IPython.display import display
import cv2
import random
from numpy import ma
from matplotlib import cbook
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec

from matplotlib.pyplot import imshow
get_ipython().magic(u'matplotlib inline')
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from pylab import rcparams

#import sys; sys.argv=['']; del sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)


# In[3]:


#### Hyper Parameters
NUM_EPOCHS = 5
BATCH_SIZE = 100
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.5
NTH_OUTPUT = 4
NTH_LAYER = 4
TARGET_LAYER = 4
NUM_NEURONS = 10
IMAGE_RANGE_255 = True


# In[4]:


#### Random seeds
torch.manual_seed(14)
torch.cuda.manual_seed_all(17)
np.random.seed(15)
random.seed(10)

torch.backends.cudnn.deterministic=True


# In[5]:


# MNIST Dataset 
train_dataset = datasets.MNIST(root='../data', 
                            train=True, 
                            transform=transforms.ToTensor(),  
                            download=True)

test_dataset = datasets.MNIST(root='../data', 
                           train=False, 
                           transform=transforms.ToTensor())

print(train_dataset, test_dataset)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=BATCH_SIZE, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=BATCH_SIZE, 
                                          shuffle=False)


# In[6]:


#### Lenet5 (Slightly Revised)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 120, kernel_size=5)
            )
        self.layer4 = nn.Sequential(
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Linear(84, 10)
        )
        
    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out3_flat = out3.view(-1, 120)
        out4 = self.layer4(out3_flat)
        out = self.layer5(out4)
        return out1, out2, out3, out4, out
    
net = Net()
net.cuda()


# In[7]:


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)  

# Train the Model
for epoch in range(NUM_EPOCHS):
    for i, (images, labels) in enumerate(train_loader):  
        # Convert torch tensor to Variable
        #images = autograd.Variable(images.view(-1, 28*28), requires_grad=True).cuda()
        if IMAGE_RANGE_255 == True:
            images = images*255
        images = autograd.Variable(images, requires_grad=True).cuda()
        labels = Variable(labels).cuda()
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(images)[NTH_OUTPUT]
        loss = criterion(outputs, labels)
        loss.backward(retain_graph=True)
        optimizer.step()
        
        if (i+1) % 600 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                   %(epoch+1, NUM_EPOCHS, i+1, len(train_dataset)//BATCH_SIZE, loss.data[0]))


# In[8]:


# Test the Model
correct = 0
total = 0

for images, labels in test_loader:
    if IMAGE_RANGE_255 == True:
        images = images*255
    images = autograd.Variable(images, requires_grad=True).cuda()
    outputs = net(images)[NTH_OUTPUT]
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


# In[9]:


#### Make test full data with all the test batches
test_full_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size=10000, shuffle=False)
for i, j in test_full_loader:
    if IMAGE_RANGE_255 == True:
        i = i*255
    test_images_full = i
    test_images_full = autograd.Variable(test_images_full, requires_grad=True).cuda()
    test_labels_full = j

print(type(test_images_full))
print(test_images_full.shape, test_labels_full.shape, test_labels_full)


# In[10]:


# Save the Model
torch.save(net.state_dict(), '/home/xlpczv/Pytorch/model/ML_model3.pth')
# Load the model
#### Don't use it carelessly
#net = torch.load('/home/xlpczv/Pytorch/model/ML_model2.pth')


# In[11]:


#### Eigenvalue decomposition function
def EVD(mat):
    mat_mean = mat.mean(dim=0)
    centered_mat = mat - mat_mean
    centered_mat_tr = torch.transpose(centered_mat, 0, 1)
    square_mat = torch.mm(centered_mat_tr, centered_mat)
    evd = torch.eig(square_mat, eigenvectors=True)
    return evd

out = net(test_images_full)[4]
mean5 = Variable(torch.unsqueeze(out.mean(dim=0), dim=0))
eigenvalue5 = torch.sort(EVD(out)[0][:,0], descending=True)[0]
eigenvalue_index5 = torch.sort(EVD(out)[0][:,0], descending=True)[1]
eigenvector5 = Variable(EVD(out)[1][:,eigenvalue_index5])

out4 = net(test_images_full)[3]
mean4 = Variable(torch.unsqueeze(out.mean(dim=0), dim=0))
eigenvalue4 = torch.sort(EVD(out4)[0][:,0], descending=True)[0]
eigenvalue_index4 = torch.sort(EVD(out4)[0][:,0], descending=True)[1]
eigenvector4 = Variable(EVD(out4)[1][:,eigenvalue_index4])

""" CHECK EVD FUNCTION

Eigenvalue decomposition
centered_out = out-out.mean(dim=0)
centered_out_tr = torch.transpose(centered_out, 0, 1)
square_out = torch.mm(centered_out_tr, centered_out)
print(square_out)


front = torch.mm(eigenvector4, torch.diag(eigenvalue4))
whole = torch.mm(front, torch.transpose(eigenvector4,0,1))
print(whole)

WELL DONE"""

""" CHECK (A-M)*EVD CORRELATION=0

centered_matrix = out - out.mean(dim=0)
new_dim_matrix = torch.mm(centered_matrix, eigenvector5).cpu().detach().numpy()
np.corrcoef(new_dim_matrix, rowvar=False)
for i in range(10):
    print(plt.hist(new_dim_matrix[:,i], alpha=0.2))
    
WELL DONE"""


# In[12]:


#### Preprocessing

#### Calculate mean and std of images by channel
images_by_channel = test_images_full.view(1, 10000, -1).contiguous().view(1, -1)
print(images_by_channel.shape)

MEAN = images_by_channel.mean(dim=1)
STD = images_by_channel.std(dim=1)
print('MEAN:', MEAN, '\n', 'STD:', STD)

#### preprocess function based on 'misc_functions.py'
def preprocess_images(images):

    # Normalize the channels
    normalized_images = images.clone()
    print('normalized_images.shape:', normalized_images.shape)

    normalized_images -= MEAN
    normalized_images /= STD
    
    #### Add one dimension at the very front
    normalized_images = torch.unsqueeze(normalized_images, dim=0)
    print(normalized_images.shape)
    
    # Convert to Pytorch variable
    normalized_images_var = Variable(normalized_images, requires_grad=True).cuda()
    return normalized_images_var


# In[13]:


#### Visualization with Color
class MidPointNorm(Normalize):
    def __init__(self, midpoint=0, vmin=None, vmax=None, clip=False):
        Normalize.__init__(self,vmin, vmax, clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if not (vmin < midpoint < vmax):
            raise ValueError("midpoint must be between maxvalue and minvalue.")      
        elif vmin == vmax:
            result.fill(0) # Or should it be all masked? Or 0.5?
        elif vmin > vmax:
            raise ValueError("maxvalue must be bigger than minvalue")
        else:
            vmin = float(vmin)
            vmax = float(vmax)
            if clip:
                mask = ma.getmask(result)
                result = ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                  mask=mask)

            # ma division is very slow; we can take a shortcut
            resdat = result.data

            #First scale to -1 to 1 range, than to from 0 to 1.
            resdat -= midpoint            
            resdat[resdat>0] /= abs(vmax - midpoint)            
            resdat[resdat<0] /= abs(vmin - midpoint)

            resdat /= 2.
            resdat += 0.5
            result = ma.array(resdat, mask=result.mask, copy=False)                

        if is_scalar:
            result = result[0]            
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if cbook.iterable(value):
            val = ma.asarray(value)
            val = 2 * (val-0.5)  
            val[val>0]  *= abs(vmax - midpoint)
            val[val<0] *= abs(vmin - midpoint)
            val += midpoint
            return val
        else:
            val = 2 * (val - 0.5)
            if val < 0:
                return  val*abs(vmin-midpoint) + midpoint
            else:
                return  val*abs(vmax-midpoint) + midpoint


# In[15]:


#### Vanilla Backprop
class VanillaBackprop():
    """
        Produces gradients generated with vanilla back propagation from the image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model._modules.items())[0][1][0]
        first_layer.register_backward_hook(hook_function)

    def generate_gradients(self, input_image, target_neuron, target_layer):
        # Forward
        model_output = self.model(input_image)[target_layer]
        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_().cuda()
        one_hot_output[0][target_neuron] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients = self.gradients.data[0]
        return gradients

    
class VanillaBackpropNewdim():
    """
        Produces gradients generated with vanilla back propagation from the image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model._modules.items())[0][1][0]
        first_layer.register_backward_hook(hook_function)

    def generate_gradients(self, input_image, target_neuron, target_layer):
        # Forward
        model_output = self.model(input_image)[target_layer]
        
        # Eigen dimension
        if target_layer == 4:
            eigenvector = eigenvector5
            mean = mean5
        elif target_layer == 3:
            eigenvector = eigenvector4
            mean = mean4
        elif target_layer == 2:
            eigenvector = eigenvector3
            mean = mean3
        elif target_layer == 1:
            eigenvector = eigenvector2
            mean = mean2
        elif target_layer == 0:
            eigenvector = eigenvector1
            mean = mean1
        else:
            raise ValueError
        
        model_output_centered = model_output - mean
        model_output_new_dim = torch.mm(model_output_centered, eigenvector)
        # NOT subtract mean
        #model_output_new_dim = torch.mm(model_output, eigenvector)
        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_().cuda()
        one_hot_output[0][target_neuron] = 1
        # Backward pass
        model_output_new_dim.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients = self.gradients.data[0]
        return gradients


# In[128]:


class run_VanillaBackprop():
    
    def __init__(self, pretrained_model, target_class, target_image):
        self.model = pretrained_model
        self.target_class = target_class
        self.target_image = target_image
        self.target_image_label = test_labels_full[target_image].item()
        self.path = '/home/xlpczv/Pytorch/pytorch-cnn-visualizations/results/ML3/'

        self.original_image = test_images_full[target_image]
        self.prep_image = preprocess_images(self.original_image)
        
    def save_original_image(self, image):
        #original_label = test_labels_full[image]
        original = test_images_full[image].cpu().detach()
        original = np.array(original, dtype=float)
        original_pixels = original.reshape((28,28))
        plt.imshow(original_pixels, cmap='gray')
        plt.savefig(self.path + 'original_image' + str(image) + "_label" + str(self.target_image_label))
    
    
    def save_gradient(self, target_layer, preprocess, new_dim, color):
        if new_dim == True:
            VBP = VanillaBackpropNewdim(self.model)
        elif new_dim == False:
            VBP = VanillaBackprop(self.model)
        else:
            raise ValueError

        if preprocess == True:
            image = self.prep_image
        elif preprocess == False:
            image = torch.unsqueeze(self.original_image, dim=0)
        else:
            raise ValueError
        
        if target_layer == 4:
            num_neurons = 10
            sizes = [30, 15, 2, 5,20, 30] # sizes = [figwidth, figheight, #rows, #cols, subfigtitlefont, figtitlefont]
        elif target_layer == 3:
            num_neurons = 84
            sizes = [30, 60, 12, 7, 40, 50]
        else:
            raise ValueError
        
        # Start making the figure 
        title = str("Vanilla new_dim_" + str(new_dim) + " layer" + str(target_layer + 1) +
                    " image" + str(self.target_image) + " label" + str(self.target_image_label) + " class " + str(self.target_class))
        fig = plt.figure(figsize=(sizes[0],sizes[1]))
        
        #### For showing all the neurons in one page
        for i in range(num_neurons):
            vanilla_grads = VBP.generate_gradients(image, i, target_layer)
            gradient = np.asarray(vanilla_grads)[0]
        
            if color == True:
                norm = MidPointNorm(midpoint=0.0)
                ax=fig.add_subplot(sizes[2],sizes[3],i+1)
                ax.set_title("neuron" + str(i), fontsize=sizes[4])
                plt.imshow(gradient, interpolation='none', norm=norm, cmap="seismic")
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(cax=cax)
                plt.clim(-0.002,0.002)
            
            elif color == False:
                gradient = gradient - gradient.min()
                gradient /= gradient.max()
                gradient = np.uint8(gradient * 255)
                
                ax=fig.add_subplot(sizes[2],sizes[3],i+1)
                ax.set_title("neuron" + str(i), fontsize=sizes[4])
                plt.imshow(gradient, interpolation='none', cmap="binary")
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(cax=cax)
            
            else:
                raise ValueError
        
        fig.suptitle("color " + title, fontsize=sizes[5])
        file_name = self.path + "ML3_color_" + title + ".jpg"
        #fig.tight_layout()
        #fig.subplots_adjust(top=0.95)
        plt.savefig(file_name)
        plt.show()
        plt.clf()


# In[129]:


for i in range(30):
    runVB = run_VanillaBackprop(pretrained_model=net, target_class="all", target_image=i)
    runVB.save_original_image(i)
    runVB.save_gradient(target_layer=4, preprocess=False, new_dim=False, color=True)
    runVB.save_gradient(target_layer=4, preprocess=False, new_dim=True, color=True)


# In[115]:


#### GuidedBackprop()
class GuidedBackprop():
    """
       #Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
            #print('grad_in.shape', grad_in[0].shape, 'grad_out.shape', grad_out[0].shape)

        # Register hook to the first layer
        first_layer = list(self.model._modules.items())[0][1][0]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            #Updates relu activation functions so that it only returns positive gradients
        """
        def relu_hook_function(module, grad_in, grad_out):
            """
            #If there is a negative gradient, changes it to zero
            """
            if isinstance(module, ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)
        # Loop through layers, hook up ReLUs with relu_hook_function
        for layer, seq in self.model._modules.items():
            for module in seq:
                if isinstance(module, ReLU):
                    module.register_backward_hook(relu_hook_function)

    def generate_gradients(self, input_image, target_neuron, target_layer):
        # Forward pass
        model_output = self.model(input_image)[target_layer]
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_().cuda()
        one_hot_output[0][target_neuron] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients = self.gradients.data[0]
        #print('gradients', gradients)
        return gradients
    
    
class GuidedBackpropNewdim():
    """
       #Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
            #print('grad_in.shape', grad_in[0].shape, 'grad_out.shape', grad_out[0].shape)

        # Register hook to the first layer
        first_layer = list(self.model._modules.items())[0][1][0]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            #Updates relu activation functions so that it only returns positive gradients
        """
        def relu_hook_function(module, grad_in, grad_out):
            """
            #If there is a negative gradient, changes it to zero
            """
            if isinstance(module, ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)
        # Loop through layers, hook up ReLUs with relu_hook_function
        for layer, seq in self.model._modules.items():
            for module in seq:
                if isinstance(module, ReLU):
                    module.register_backward_hook(relu_hook_function)

    def generate_gradients(self, input_image, target_neuron, target_layer):
        # Forward pass
        model_output = self.model(input_image)[target_layer]
        #print('model_output', model_output)
        # Eigen dimension
        if target_layer == 4:
            eigenvector = eigenvector5
            mean = mean5
        elif target_layer == 3:
            eigenvector = eigenvector4
            mean = mean4
        elif target_layer == 2:
            eigenvector = eigenvector3
            mean = mean3
        elif target_layer == 1:
            eigenvector = eigenvector2
            mean = mean2
        elif target_layer == 0:
            eigenvector = eigenvector1
            mean = mean1
        else:
            raise ValueError
        
        model_output_centered = model_output - mean
        model_output_new_dim = torch.mm(model_output_centered, eigenvector)
        # Not subtract mean
        #model_output_new_dim = torch.mm(model_output, eigenvector)
        #print('model_output_new_dim', model_output_new_dim)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_().cuda()
        one_hot_output[0][target_neuron] = 1
        # Backward pass
        model_output_new_dim.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients = self.gradients.data[0]
        #print('new dim gradients', gradients)
        return gradients


# In[116]:


class run_GuidedBackprop():
    
    def __init__(self, pretrained_model, target_class, target_image):
        self.model = pretrained_model
        self.target_class = target_class
        self.target_image = target_image
        self.target_image_label = test_labels_full[target_image]
        self.path = '/home/xlpczv/Pytorch/pytorch-cnn-visualizations/results/ML3/'

        self.original_image = test_images_full[target_image]
        self.prep_image = preprocess_images(self.original_image)
    
    def save_original_image(self, image):
        original_label = test_labels_full[image]
        original = test_images_full[image].cpu().detach()
        original = np.array(original, dtype=float)
        original_pixels = original.reshape((28,28))
        plt.imshow(original_pixels, cmap='gray')
        plt.savefig(self.path + 'original_image' + str(image) + "_label" + str(original_label))
    
    def save_gradient(self, target_layer, preprocess, new_dim, color):
        if new_dim == True:
            GBP = GuidedBackpropNewdim(self.model)
        elif new_dim == False:
            GBP = GuidedBackprop(self.model)
        else:
            raise ValueError

        if preprocess == True:
            image = self.prep_image
        elif preprocess == False:
            image = torch.unsqueeze(self.original_image, dim=0)
        else:
            raise ValueError
        
        if target_layer == 4:
            num_neurons = 10
            sizes = [30, 15, 2, 5,20, 30] # sizes = [figwidth, figheight, #rows, #cols, subfigtitlefont, figtitlefont]
        elif target_layer == 3:
            num_neurons = 84
            sizes = [30, 60, 12, 7, 20, 30]
        else:
            raise ValueError

        
        # Start making the figure
        fig = plt.figure(figsize=(sizes[0],sizes[1]))
        title = str("Guided new_dim_" + str(new_dim) + " layer" + str(target_layer + 1) +
                    " image" + str(self.target_image) + " label" + str(self.target_image_label.item()) + " class " + str(self.target_class))
        
        #### For showing all the neurons in one page
        for i in range(num_neurons):
            guided_grads = GBP.generate_gradients(image, i, target_layer)
            gradient = np.asarray(guided_grads)[0]

            if color == True:
                norm = MidPointNorm(midpoint=0.0)
                ax=fig.add_subplot(sizes[2],sizes[3],i+1)
                ax.set_title("neuron" + str(i), fontsize=sizes[4])
                plt.imshow(gradient, interpolation='none', norm=norm, cmap="seismic")
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(cax=cax)
                plt.clim(-0.002,0.002)
            
            else:
                gradient = gradient - gradient.min()
                gradient /= gradient.max()
                gradient = np.uint8(gradient * 255)
                
                ax=fig.add_subplot(sizes[2],sizes[3],i+1)
                ax.set_title("neuron" + str(i), fontsize=sizes[4])
                plt.imshow(gradient, interpolation='none', cmap="binary")
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(cax=cax)
                fig.suptitle("Guided black   " + title, fontsize=sizes[5])
                file_name = self.path + "ML3_Guided_black_" + title + ".jpg"
        
        fig.suptitle("color " + title, fontsize=sizes[5])
        file_name = self.path + "ML3_color_" + title + ".jpg"
        fig.tight_layout()
        fig.subplots_adjust(top=0.95)
        plt.savefig(file_name)
        plt.show()
        plt.clf()


# In[119]:


runGB = run_GuidedBackprop(pretrained_model=net, target_class="all", target_image=100)
runGB.save_original_image(100)
runGB.save_gradient(target_layer=4, preprocess=False, new_dim=False, color=True)

runGB = run_GuidedBackprop(pretrained_model=net, target_class="all", target_image=100)
runGB.save_gradient(target_layer=4, preprocess=False, new_dim=True, color=True)

