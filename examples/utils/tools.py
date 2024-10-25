import torch, random, os
import numpy as np
import copy
from torch.autograd import Variable

def factorial(x):
    result = 1
    for i in range(2, x + 1):
        result *= i
    return result

def adjust_learning_rate(optimizer, lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr
    return lr


def init_random_seed(manual_seed, cuda=None):
    '''
    if cuda!=None:
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda
    elif torch.cuda.device_count()>=1:
        devices = [str(i) for i in range(torch.cuda.device_count())]
        np.random.shuffle(devices)
        os.environ['CUDA_VISIBLE_DEVICES']=",".join(devices)
    
    print("os.environ['CUDA_VISIBLE_DEVICES']: ",
          os.environ['CUDA_VISIBLE_DEVICES'])
    '''
    """Init random seed."""
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    #print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
        
    torch.backends.cudnn.deterministic = True
    # if False, sacrifice the computation efficiency
    torch.backends.cudnn.benchmark = False
        
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
           
def emd(net_glob_out, net_out, label=None):
    # Manhattan Distance
    #dist = torch.sum(torch.sum(torch.abs(net_glob_out - net_out), 1))/len(net_glob_out)
    label_data = dict([(int(i),[None, None]) for i in torch.unique(label)])
    
    for l, i, j  in zip(label.data, net_glob_out, net_out):
        if type(label_data[int(l)][0]) == type(None):
            label_data[int(l)][0] = i
            label_data[int(l)][1] = j.reshape((1,-1))
        else:
            label_data[int(l)][0] += i
            label_data[int(l)][1] = torch.cat((label_data[int(l)][1], 
                                               j.reshape((1,-1))), 0)
            
    dist = []
    for no, (l, [i, j]) in enumerate(label_data.items()):
        global_center = i/len(j)
        dist.append(torch.sum(torch.sum(torch.abs(j - global_center), 1))/len(j))
    dist = sum(dist)/len(dist)
    return dist


def cal_weight_diff(local_weights, avg_weights):
    w = copy.deepcopy(local_weights)
    w2 = copy.deepcopy(avg_weights)

    # List for differences: for all devices, get the average of abs((val(device)-val(average))/val(average))
    diff_list = []  
    #print("\n\tWeight difference:")

    for key in list(w2.keys()):
        average = torch.sum((w[key]-w2[key])**2)
        diff_list.append(average) 

    return sum(diff_list)

def cal_avg_weight_diff(weights_list, avg_weights):
    w = copy.deepcopy(weights_list)
    w2 = copy.deepcopy(avg_weights)

    key = list(w2.keys())

    for key in list(w2.keys()):
        w2[key] = w2[key].reshape(-1).tolist()    # Reshape to 1d tensor and transform to list

    # List for differences: for all devices, get the average of abs((val(device)-val(average))/val(average))
    diff_list = []  
    print("\n\tWeight difference:")

    for key in list(w2.keys()):
        tmp2 = []
        for i in range(len(w)):
            tmp = []
            w[i][key] = w[i][key].reshape(-1).tolist()    # Reshape to 1d tensor and transform to list

            for j in range(len(w[i][key])):
                tmp.append(abs((w[i][key][j]-w2[key][j])/w2[key][j]))   # Abs((val(device)-val(average))/val(average))

            average = sum(tmp)/len(tmp) # Calculate average
            tmp2.append(average) 
            print(f"\t\tWeight difference | Weight {i + 1} | {key} | {average}")

        average = sum(tmp2)/len(tmp2) # Calculate average
        diff_list.append(average) 

    return sum(diff_list)/len(diff_list)

def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))
    

def print_model_parm_flops(model, input_shape):

    prods = {}
    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)
        return hook_per

    list_1=[]
    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))
    list_2={}
    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)


    multiply_adds = False
    list_conv=[]
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_conv.append(flops)


    list_linear=[] 
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn=[] 
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())

    list_relu=[] 
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling=[]
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_pooling.append(flops)



    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            return
        for c in childrens:
                foo(c)

    foo(model)
    input = Variable(torch.rand(input_shape).unsqueeze(0), requires_grad = True)
    _ = model(input)


    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))

    print('  + Number of FLOPs: %.2fM' % (total_flops / 1e6))
    
def dp_noise(shape, sigma):
    noise = sigma*torch.randn(shape)#torch.FloatTensor(param.shape).normal_(mean=0, std=sigma)
    return noise

def model_dist_norm(model, target_params):
    squared_sum = 0
    param_dict = dict(model.state_dict())
    for name, layer in param_dict.items():
        if 'running_' in name or '_tracked' in name:
            continue
        squared_sum += torch.sum(torch.pow(layer.data - \
                                           target_params[name].clone().detach().requires_grad_(False).data, 2))
    return torch.sqrt(squared_sum)