# 鲁迅说没说
## 1. 项目概述
输入一句疑似鲁迅说过的名言，本项目会在鲁迅作品集中搜索多个语义相近的句子，输出给大模型，大模型会判断用户输入的疑似名言与搜索结果的相似程度，判断疑似名言是否出自鲁迅作品集。
鲁迅作品集来自[luxun_dataset](https://github.com/sun510001/luxun_dataset)。 在原数据的基础上，设置以下字段："book"、"title"、"author"、"type"、"source"、"date"和"content"，但是因为在原数据的基础上整理，有些字段的值为空。整理后的数据请见[鲁迅作品数据集](https://github.com/BushJiang/LuXun_dataset)。


## 2. 安装
### 2.1 安装 Docker
Milvus 运行在 Docker 容器中，因此需要先安装 Docker Desktop。

Milvus 运行在 docker 容器中，所以需要先安装 Docker Desktop。
MacOS 系统安装方法：[Install Docker Desktop on Mac](https://docs.docker.com/desktop/install/mac-install/)。
Windows 系统安装方法：[Install Docker Desktop on Windows](https://docs.docker.com/desktop/install/windows-install/)。

### 2.2 安装向量数据库 Milvus
下载并运行 Milvus 独立版脚本：

下载安装脚本：
```shell
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
```

运行 Milvus：
```shell
bash standalone_embed.sh start
```

### 2.3 创建 python 虚拟环境（可选）
建议创建虚拟环境以避免依赖冲突：

```shell
python3 -m venv myenv
```

激活虚拟环境
```shell
source myenv/bin/activate
```

退出虚拟环境：
```shell
deactivate
```

### 2.4 安装 Python 依赖
```shell
pip install -r requirements.txt
```

### 2.5 获取大模型的 api key
本项目采用 deepseek 的 api key，可以在[这里](https://platform.deepseek.com/api_keys)获取。
然后在终端输入命令：`export DEEPSEEK_API_KEY=<your-deepseek-api-key>`，把deepseek的api key作为环境变量保存。把`<your-deepseek-api-key>`替换为你获取的 api key。

### 2.6 配置文件
本项目的配置文件为`config.yaml`，其中`limit`是向量数据库搜索结果的数量，增加该数值可以搜索到更多语义相似的句子，但是会增加搜索时间，以及输入给大模型的 token 数量。

## 3. 运行项目
执行 main.py 交互程序。
```shell
python main.py
```

支持以下四种命令：
`create <file_path>`：在向量数据库中创建集合，生成指定文本的向量，导入集合中，并且创建索引。把`<file_path>`替换成文件路径。比如`create data/LuXunWorks.json>`。本项目只提供了 LuXunWorks.json 文件，你也可以替换成其他文件，但是文件格式必须相同。

`ask`：进入问答模式。在问答模式中，输入疑似鲁迅的名言，项目会判断鲁迅是否说过这句话。比如：

```shell
（ask）请输入疑似鲁迅的名言：浪费别人的时间是谋财害命，浪费自己的时间是慢性自杀
鲁迅的确说过类似的话，原文是“美国人说，时间就是金钱；但我想：时间就是性命。无端的空耗别人的时间，其实是无异于谋财害命的。不过像我们这样坐着乘风凉，谈闲天的人们，可又是例外。”，这句话来自《门外文谈》。
```

`delete`：删除已有集合。

`exit`：退出当前命令，或者退出项目。

## 4. 许可证
本项目采用 MIT 许可证。