import tensorflow_hub as hub
import numpy as np
import tensorflow as tf
import os
import csv
import time

import torch
import tokenization
import networkx as nx
from transformers import BertTokenizer, BertModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定 GPU 设备编号，如有多张 GPU，可以设置 "0", "1", etc.

# 确保 TensorFlow 使用 GPU 进行计算
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # 避免 GPU 内存一次性分配完
        print("GPU is ready for TensorFlow!")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Please check your TensorFlow installation.")

edgetype2id = {'JUMPI-True': 0, 'JUMPI-False': 1, 'JUMP': 2, 'Sequence': 3, 'CALLPRIVATE': 4, 'RETURNPRIVATE': 5,
               'CALL': 6, 'STATICCALL': 7, 'DELEGATECALL': 8, 'RETURN': 9, 'DATAFLOW': 10}

def create_info(path):
    # embed = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/large/2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for address in os.listdir(path):
        project_path = os.path.join(path, address)
        for project in os.listdir(project_path):
            node2id = {}
            idx = 0
            datapath = os.path.join(project_path, project)
            t_start = time.time()
            # print(datapath)
            result_path = f'E:/Py_projects/CrossVulDec/utils/inputdata_768/{address}/{project}'
            # if os.path.exists(result_path) is False:
            #     os.makedirs(result_path)
            '''
            # 判断特征维度是不是768
            flag = True
            if os.path.exists(os.path.join(datapath, 'graph_node.txt')):
                stat = os.stat(os.path.join(datapath, 'graph_node.txt'))
                mod_time = stat.st_mtime
                # 将时间戳转换为本地时间
                local_time = time.localtime(mod_time)
                # 格式化时间为可读的形式
                formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', local_time)
                modified_day = formatted_time.split(' ')[0]
                if modified_day != '2024-09-24' and modified_day != '2024-09-25':
                    flag = False
                    print(f"文件 '{os.path.join(datapath, 'graph_node.txt')}' 的最后修改时间是: {formatted_time}")
            if flag:
                continue
            '''
            with open(os.path.join(datapath, 'graph_node.txt'), 'w', encoding='utf-8') as f:
                # csvwriter = csv.writer(f)
                # with open(os.path.join(datapath, f'{project}_node_attr.txt'), 'w', encoding='utf-8') as attr_f:
                for file in os.listdir(datapath):
                    if file.endswith('.gpickle'):
                        # print(os.path.join(datapath, file))
                        # print(os.path.join(datapath, file))
                        g = nx.read_gpickle(os.path.join(datapath, file))
                        for node in g.nodes:
                            embed = hub.load(
                                "https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/large/2")
                            # 一个项目中几乎除了0x0没有重复的节点
                            node_id = node  # ERC20_0xab_node
                            if node == '0x0':
                                node_id = f"{project}_0x0"
                            # if 'return' in node:
                            #     print("RETURN node: " + node)
                            if node_id not in node2id.keys():
                                node2id[node_id] = idx
                                idx += 1
                                nodetype = g.nodes[node]['type_list']
                                nodeattr = g.nodes[node]['attr']
                                f.write(str(node2id[node_id])+'\t')
                                flag = True
                                for type_ in nodetype:
                                    if flag:
                                        flag = False
                                        f.write(type_)
                                    else:
                                        f.write(',' + type_)
                                # f.write('\t'+nodeattr+'\t')
                                ##################################
                                #   nodeattr没有必要写入，节省空间   #
                                #################################
                                # print(type(nodeattr))
                                text = [nodeattr]
                                # print(text)
                                feats = embed(text)
                                list_data = tf.squeeze(feats).numpy().tolist()
                                # print(list_data)
                                for i in range(len(list_data)):
                                    list_data[i] = '{:.6f}'.format(list_data[i])
                                formatted_row = ','.join(map(str, list_data))
                                # for row in list_data:
                                #     formatted_row = ','.join(['{:.6f}'.format(num) for num in row])
                                f.write('\t'+formatted_row + '\n')
                                # attr_f.write(nodeattr+'\n')
                print(f"{datapath} is ok!")
            with open(os.path.join(datapath, 'graph_link.txt'), 'w', encoding='utf-8') as f:
                for file in os.listdir(datapath):
                    if file.endswith('.gpickle'):
                        g = nx.read_gpickle(os.path.join(datapath, file))
                        for u, v, type_ in g.edges(data=True):
                            if u == '0x0' or v == '0x0':
                                # print(f"{u} -> {v}")
                                continue
                            else:
                                f.write(f"{node2id[u]}\t{node2id[v]}\t{edgetype2id[type_['type']]}\t{1.0}\n")
            t_end = time.time()
            print('Time(s) {:.4f}'.format(t_end - t_start))


create_info('E:\\Py_projects\CrossVulDec\\vul_data\\hidden_768\\cross-contract\\USE\\cross-reentrancy-Slise') #记得更换result_path <-> data_path
# create_info('E:\\Py_projects\\DataFlow\\cross-contract\\cross-integeroverflow')
# create_info('../vul_data/hidden_768/cross-contract/cross-access_control/SmartState_bytecode')
# for i in range(16, 31):
#     create_info(f'../data_{i}')
# 之后使用Bert将node_attr转换成embedding
# 将embedding与graph_node合并，得到GNN最终的输入数据


