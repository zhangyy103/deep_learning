- 清屏
```
cls
```  
---
---
- 列出所有的环境  
```
conda env list  
```
- 创建名为“环境名”的虚拟环境，并指定 Python 的版本  
```
conda create -n 环境名 python=3.9  
```
- 创建名为“环境名”的虚拟环境，并指定 Python 的版本与安装路径  
```
conda create --prefix=安装路径\环境名 python=3.9  
```
- 删除名为“环境名”的虚拟环境  
```
conda remove -n 环境名 --all  
```
- 进入名为“环境名”的虚拟环境  
```
conda activate 环境名  
```
---
---
- 列出当前环境下的所有库  
```
conda list  
```
- 安装 NumPy 库，并指定版本 1.21.5  
```
pip install numpy==1.21.5 -i https://pypi.tuna.tsinghua.edu.cn/simple  
```
- 安装 Pandas 库，并指定版本 1.2.4  
```
pip install Pandas==1.2.4 -i https://pypi.tuna.tsinghua.edu.cn/simple  
```
- 安装 Matplotlib 库，并指定版本 3.5.1  
```
pip install Matplotlib==3.5.1 -i https://pypi.tuna.tsinghua.edu.cn/simple  
```
- 查看当前环境下某个库的版本（以 numpy 为例）  
```
pip show numpy  
```
- 退出虚拟环境  
```
conda deactivate  
```