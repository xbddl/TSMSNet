
class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = "norm"
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'LEVIR':
            self.root_dir = 'E:/swc/data/LEVIR-CD'

        elif data_name == 'WHU':
            self.root_dir = 'E:/swc/data/WHU-CD'  #'../WHU-CD'

        elif data_name == 'GVLM':
            self.root_dir = 'E:/swc/data/GVLM-CD'

        elif data_name == 'Google':
            self.root_dir = 'E:/swc/data/CD_Data_GZ'
            
        elif data_name == 'HGG':
            self.root_dir = 'E:/swc/data/GVLM_Google-CD'
            
        elif data_name == 'DSIFN':
            self.root_dir = 'F:/CD_Dataset/DSIFN-Dataset_256'
            
        elif data_name == 'HRCUS':
            self.root_dir = 'E:/swc/data/HRCUS-CD'

        elif data_name == 'quick_start':
            self.root_dir = './samples/'
        else:
            raise TypeError('%s has not defined' % data_name)
        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='LEVIR')
    print(data.data_name)
    print(data.root_dir)
    print(data.label_transform)

