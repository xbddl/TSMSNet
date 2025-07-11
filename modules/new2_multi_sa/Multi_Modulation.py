import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class PredictMap(nn.Module):
    def __init__(self, dim,class_num=2):
        super().__init__()
        self.conv1=nn.Conv2d(dim,dim//2,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(dim//2)
        self.conv2=nn.Conv2d(dim//2,32,kernel_size=3,stride=1,padding=1)
        self.conv3=nn.Conv2d(32,class_num,kernel_size=3,stride=1,padding=1)
    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=F.relu(x)
        x=self.conv2(x)
        x=self.conv3(x)
        pred = F.softmax(x, dim=1)
        return x,pred

class Multi_Modulation(nn.Module):
    def __init__(self, dim,scales=3,class_num=2):
        super().__init__()
        self.scales = scales
        self.w = nn.Conv2d(dim, dim, 1)
        self.bn1 = nn.BatchNorm2d(dim)
        self.modulation = nn.ModuleList([nn.Sequential(nn.Conv2d(dim, dim, kernel_size=(11 - i * 2), stride=1, padding=(5 - i), groups=dim), nn.BatchNorm2d(dim))
                                         for i in range(scales)
                                    ])
        self.conv2 = nn.ModuleList([nn.Conv2d(dim, dim, 1),
                                    nn.Sequential(nn.Conv2d(dim, dim,
                                                            kernel_size=3, stride=1, padding=(1),groups=dim),
                                                  nn.BatchNorm2d(dim),
                                                  nn.Conv2d(dim, dim,
                                                            kernel_size=(3, 1), stride=1, padding=(1, 0),groups=dim),
                                                  nn.BatchNorm2d(dim),
                                                  nn.Conv2d(dim, dim,
                                                            kernel_size=(1, 3), stride=1, padding=(0, 1),groups=dim),
                                                  nn.BatchNorm2d(dim)),
                                    nn.Sequential(nn.Conv2d(dim, dim,
                                                            kernel_size=5, stride=1, padding=(2),groups=dim),
                                                  nn.BatchNorm2d(dim),
                                                  nn.Conv2d(dim, dim,
                                                            kernel_size=(5, 1), stride=1, padding=(2, 0),groups=dim),
                                                  nn.BatchNorm2d(dim),
                                                  nn.Conv2d(dim, dim,
                                                            kernel_size=(1, 5), stride=1, padding=(0, 2),groups=dim),
                                                  nn.BatchNorm2d(dim))
                                    ])
        
        self.shortcut = nn.Identity()


    def forward(self, x):
        B, C, H, W = x.shape
        fea = self.w(x)
        fea = self.bn1(fea)
        shortcut = self.shortcut(x)
        
        # output0 = self.modulation[0](fea)*self.conv2[0](fea)
        # output1 = self.modulation[1](fea)*self.conv2[1](fea)
        # output2 = self.modulation[2](fea)*self.conv2[2](fea)
        
        # predfeature,predictmap = self.predictmap(x)

        # change_map = predictmap[:,1:2,:,:]
        # pred_gate0 = change_map
        # pred_gate1 = change_map
        # pred_gate2 = change_map
        
        # modulation0 = self.modulation[0](fea) * pred_gate0
        # modulation1 = self.modulation[1](fea) * pred_gate1
        # modulation2 = self.modulation[2](fea) * pred_gate2
        
        
        
        # output0 = modulation0 * self.conv2[0](fea)
        # output1 = modulation1 * self.conv2[1](fea)
        # output2 = modulation2 * self.conv2[2](fea)
        
        # output = torch.cat([output0,output1,output2],dim=1)
        
        # output = self.conv3(output)
        # output = output + shortcut
        #output = self.modulation[0](fea)*self.conv2[0](fea.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        output = self.modulation[0](fea)*self.conv2[0](fea)
        for i in range(1,self.scales):
            output = output + self.modulation[i](fea)*self.conv2[i](fea)
        output = output + shortcut
        return output

                                                                          
class Multi_Modulation_Block(nn.Module):
    def __init__(self, dim,scales=3,class_num=2,status=True): 
        super().__init__()
        self.scales = scales
        self.modulations = Multi_Modulation(dim,scales)
        #self.conv1 = nn.Conv2d(dim, dim, 1)
        self.shortcut = nn.Identity()
        self.predictmap = PredictMap(dim,class_num)
        #self.conv2 = nn.Conv2d(dim, dim, 1)
        self.status = status
        
    def forward(self, x):
        B, C, H, W = x.shape
        fea = self.modulations(x)
        shortcut = self.shortcut(fea)
        #fea = self.conv1(fea)
        if self.status:
            predfeature,predictmap = self.predictmap(fea)
            pred_change = predictmap[:,1:2,:,:]
            output = fea*pred_change + shortcut
            #output = self.conv2(output)
            return output,predfeature
        else:
            output = fea
            return output,None


# if __name__ == "__main__":
#     x = torch.randn(4, 64, 128, 128)
#     model1 = Multi_Modulation_Block(dim=64,class_num=2)
#     output1,predictmap = model1(x)
#     print(output1.size())
