import pandas as pd  
  
import pandas as pd  
  
def write_list_to_excel(data_list, file_path, sheet_name='Sheet1'):  
    # 创建一个DataFrame，data_list的每个内部列表是一行数据  
    df = pd.DataFrame(data_list)  
      
    # 将DataFrame写入Excel文件，指定sheet名称  
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:  
        df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)  
    
    