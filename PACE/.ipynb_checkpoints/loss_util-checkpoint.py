  
import torch.nn as nn
import torch.nn.functional as F
import torch

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    def forward(self, output1, output2, target, use_domain, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)+0.001  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - distances.sqrt()).pow(2))
        losses = losses*use_domain
        return losses.mean() if size_average else losses.sum()
    
    

class WassersteinDistance(nn.Module):
    """
    2-Wasserstein distance between two distributions
    https://en.wikipedia.org/wiki/Wasserstein_metric#Normal_distributions
    """
    def __init__(self):
        
        super(WassersteinDistance, self).__init__()
        
    def forward(self, mu1, mu2, sigma1, sigma2, reduction='mean'):
        factor1 = torch.norm(mu1-mu2)
        factor2 = sigma1+sigma2 - 2*torch.pow( (torch.pow(sigma2, 0.5) * sigma1 * torch.pow(sigma1, 0.5)), 0.5)
        factor2 = factor2.sum(dim=1) #applying trace on each row. each row of the sigma is the diagonal of the covariance matrix for each sample
        
        output = factor1+factor2
        if reduction == 'mean':
            output = output.mean()
        if reduction == 'max' :
            output = output.max()
            
        return output


#from: https://github.com/ZongxianLee/MMD_Loss.Pytorch/blob/master/mmd_loss.py
class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return
    
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)
    
    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss