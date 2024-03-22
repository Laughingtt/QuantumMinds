<!-- TOC -->

    * [1.模型环境搭建](#1模型环境搭建)
    * [2. 模型安装下载](#2-模型安装下载)
    * [3. 向量数据库安装](#3-向量数据库安装)
    * [4. 知识图谱安装部署](#4-知识图谱安装部署)

<!-- TOC -->

### 1.模型环境搭建

**conda 安装**

conda创建环境

```shell
conda create -n py310 python=3.10
```

查看安装环境

```shell
conda env list
```

激活虚拟环境

```shell
conda activate pyenv
```

拉取本项目代码

```shell
git clone https://github.com/Laughingtt/QuantumMinds.git
```

安装环境依赖

```shell
cd QuantumMinds 
pip install  -r requirements.txt
```

### 2. 模型安装下载

下载官方Intern_LLM模型，配置模型路径

```shell
python model/get_intern_llm.py 
```

也可以下载 知心大姐-唠五毛模型

```shell
mkdir laowumao && laowumao
#请确保已安装 git-lfs
git lfs install
#下载当前分支
git clone https://code.openxlab.org.cn/DD-learning/InternLM-LaoWuMao.git
```

### 3. 向量数据库安装

下载知识库及 embedding模型

```shell
cd InternLM-LaoWuMao/knowledge_vector_base
# 解压知识库书籍
unzip knowledge.zip -d .
# 解压embedding模型
unzip bce-embedding-base_v1.zip -d .
```

配置模型,知识库等路径

```shell
model/load_chain.py
```

### 4. 知识图谱安装部署

获取neo4j相关安装包

```shell
cd InternLM-LaoWuMao/knowledge_graph_base

# 获取知识图安装包
apoc-5.16.1-core.jar	
jdk-21_linux-x64_bin.tar.gz	
neo4j-community-5.16.0-unix.tar.gz	
neo4j.conf
```

环境安装步骤参考

&nbsp;
[KGQA部署.md](KGQA%B2%BF%CA%F0.md)

构建知识库

```shell
cd submodule/KGQA-Psychological-Counseling/data/
python build_graph.py
```

### 5.模型启动

```shell
bash run.sh
```