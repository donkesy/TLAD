import re
import os
import shutil
import networkx as nx
import numpy as np
import pandas as pd
from graphviz import Digraph
import matplotlib.pyplot as plt

'''
对使用gigahorse生成的三地址表示内容进行分割，将文件中的每个块定义为一个Block对象
'''
vis_block = []
callprivate = {}
calledprivate = []
function_selector = True


class Block: #目前缺少节点属性，之后要加上！
    def __init__(self, block_number, prev, succ, instructions):
        self.block_number = block_number
        self.prev = prev.split(', ')
        self.prev = [item for item in self.prev if item != '']
        self.succ = succ.split(', ')
        self.succ = [item for item in self.succ if item != '']
        self.instructions = instructions
        ## 用来处理循环结构
        self.in_degree = len(self.prev)
        self.out_degree = len(self.succ)

    def __repr__(self):
        return f"Block({self.block_number}, Prev: {self.prev}, Succ: {self.succ})"


def parse_blocks(file_content):
    blocks = {}
    block_pattern = "Begin block"
    instruction_pattern = "================================="
    current_block = None
    funcName = None
    functions = {}
    block_num = 0
    prev = succ = ''

    file_content = file_content.splitlines()

    for index, line in enumerate(file_content):
        if block_pattern in line:
            if current_block is not None:
                # 添加上一个 block 到列表
                blocks[block_num] = current_block
            # 创建新的 Block 对象
            block_num = line.strip().split()[-1]
            # print(file_content[index+1])
            preandsuc = file_content[index + 1].strip()
            match = re.match(r"prev=\[(.*?)\], succ=\[(.*?)\]", preandsuc)
            prev = match.group(1)  # 假设每个 block 只有一个前驱，实际可能需要根据上下文解析
            succ = match.group(2)
        elif instruction_pattern in line:
            # 收集指令
            instructions = []
            cur_index = index + 1
            while True and cur_index < len(file_content):
                line = file_content[cur_index].strip()
                if line.strip() == "":
                    break
                instructions.append(line)
                cur_index += 1
            current_block = Block(block_number=block_num, prev=prev, succ=succ, instructions=instructions)
            new_instructions = []
            new_block = False
            for i in range(len(instructions)):
                call_instruction = instructions[i].split(' ')
                new_instructions.append(instructions[i])
                if len(call_instruction) < 4:
                    continue
                if call_instruction[3] == 'CALL' or call_instruction[3] == 'STATICCALL' or call_instruction[3] == 'DELEGATECALL':
                    new_block_number = f"{block_num}_return"
                    functions[funcName].append(Block(block_number=block_num, prev=prev, succ=new_block_number, instructions=new_instructions))
                    blocks[block_num] = Block(block_number=block_num, prev=prev, succ=new_block_number, instructions=new_instructions)
                    new_instructions = []
                    new_block = True
                    block_num = f'{block_num}_return'
            if new_block:
                current_block = Block(block_number=f'{block_num}', prev=block_num, succ=succ, instructions=new_instructions)
                # block_num = f'{block_num}_return'
                functions[funcName].append(current_block)
            else:
                functions[funcName].append(current_block)
        elif "function" in line:
            funcName = line.split(' ')[1]
            functions[funcName] = []

    if current_block is not None:
        # blocks.append(current_block)  # 添加最后一个 block
        blocks[block_num] = current_block
        functions[funcName].append(current_block)
    return functions, blocks


def get_info(path):
    #  tac 文件的路径
    # path = "../tacDir/ABIEncode.tac"
    with open(path, 'r', encoding='utf-8') as f:
        file_content = f.read()

    functions, blocks = parse_blocks(file_content)

    # print(blocks)
    # print(functions)

    return functions, blocks


# print(blocks)
# print(functions)
# print(blocks['0x0'].succ)
# print(blocks['0x0'].instructions)

# skip = blocks['0x0'].instructions[-1].split(' ')
# print(skip)
########################################################################
#                                                                      #
#                          为每个函数构建CFG                              #
#                                                                      #
########################################################################
skip_instructions = ['JUMP', 'JUMPI', 'CALLPRIVATE', 'RETURN', 'REVERT']  # 如果跳转指令不在这里面，则是顺序连接——Sequence


def determine_edge_type(skip, instruction, nxt_block_num):
    edge_type = ''
    if skip == 'JUMP':
        edge_type = 'JUMP'
    elif skip == 'JUMPI':
        nxt = re.search(r'\(([^)]*)\)', instruction[2]).group(1)
        if nxt in nxt_block_num:
            edge_type = 'JUMPI-True'
        else:
            edge_type = 'JUMPI-False'
    elif 'v' in skip and instruction[-1] == 'CONST':
        edge_type = 'Sequence'
    else:
        edge_type = 'Sequence'
    return edge_type


def update_graph(blocks, skip, instruction, nxt_block_num, block_num, G):
    edge_type = determine_edge_type(skip, instruction, nxt_block_num)
    nxt_block = blocks[nxt_block_num]
    G.add_node(nxt_block.block_number, label=nxt_block.block_number, attr=nxt_block.instructions)
    G.add_edge(block_num, nxt_block_num, type=edge_type)
    nxt_block.in_degree -= 1
    blocks[block_num].out_degree -= 1


def update_callprivate(entry, instruction, block_num):
    ####################
    # return
    global vis_block, callprivate, calledprivate
    if 'CALLPRIVATE' in instruction and instruction[1] == 'CALLPRIVATE':  ##CALLPRIVATE 的参数可能不止一个
        # 0x9a1: CALLPRIVATE v99e(0xb6d), v434, v42a, v99b, v998(0x9a2)
        if '(' not in instruction[2]:
            return
        nxt = re.search(r'\(([^)]*)\)', instruction[2]).group(1)
        callprivate[entry].append({block_num: nxt})
        calledprivate.append(nxt)
    elif 'CALLPRIVATE' in instruction and instruction[3] == 'CALLPRIVATE':
        if '(' not in instruction[4]:
            return
        nxt = re.search(r'\(([^)]*)\)', instruction[4]).group(1)
        callprivate[entry].append({block_num: nxt})
        calledprivate.append(nxt)
    elif 'CALLPRIVATE' in instruction and instruction[4] == 'CALLPRIVATE':
        if '(' not in instruction[5]:
            return
        nxt = re.search(r'\(([^)]*)\)', instruction[5]).group(1)
        callprivate[entry].append({block_num: nxt})
        calledprivate.append(nxt)


def construct_cfg(entry, blocks, block_num, G, parent):
    global vis_block, callprivate

    Succ = blocks[block_num].succ

    if len(blocks[block_num].instructions) != 0:
        instruction = blocks[block_num].instructions[-1].split(' ')
        skip = instruction[1]
    else:
        instruction = ['Transition Node']
        skip = 'None'

    if len(Succ) == 0 and entry != '0x0':
        if 'CALLPRIVATE' in instruction:
            if function_selector:
                return
            update_callprivate(entry, instruction, block_num)
        # elif 'RETURNPRIVATE' in instruction:
        #     G.add_edge(block_num, parent, type='RETURNPRIVATE')
    elif len(Succ) != 0 and 'CALLPRIVATE' in instruction and entry != '0x0':
        if function_selector:
            return
        update_callprivate(entry, instruction, block_num)

        for nxt_block_num in Succ:
            if blocks[nxt_block_num].in_degree == 0:
                continue
            update_graph(blocks, skip, instruction, nxt_block_num, block_num, G)
            vis_block.append(nxt_block_num)
            construct_cfg(entry, blocks, nxt_block_num, G, parent)
    else:
        for nxt_block_num in Succ:
            if blocks[nxt_block_num].in_degree == 0:
                continue
            update_graph(blocks, skip, instruction, nxt_block_num, block_num, G)
            vis_block.append(nxt_block_num)
            construct_cfg(entry, blocks, nxt_block_num, G, parent)


########################################################################
#               从0x0块递归，构建一个完整全局的CFG                          #
########################################################################
def construct_full_cfg(path):
    funcs, blocks = get_info(os.path.join(path, 'contract.tac'))

    start_block = '0x0'

    G = nx.DiGraph()
    block = blocks[start_block]
    G.add_node(start_block, label=block.block_number, attr=block.instructions)
    construct_cfg(blocks, start_block, G, parent=start_block)

    #############################
    #      全局CFG可视化         #
    ############################
    """
    dot = Digraph('G', filename='data/delegatecall/cfg_delegatecall.gv')
    dot.attr(style='filled', color='lightgrey', rankdir='TB', overlap='scale', splines='polyline', ratio='fill')
    dot.attr('node', shape='rectangle', style='filled', fillcolor='grey',
             gradientangle='360', label='n9:360', fontcolor='black')
    dot.node_attr.update(style='filled', color='white')

    for block in blocks.values():
        dot.node(block.block_number, label=block.block_number)

    for _from, _to in G.edges:
        # print(f'{_from}->{_to}')
        dot.edge(_from, _to, color='blue')

    dot.render(filename='data/delegatecall/cfg_delegatecall.gv', view=True)
    dot.view()
    """


# construct_full_cfg()

'''
pos = nx.spectral_layout(G)  # 布局算法
# nx.draw(G, pos, with_labels=True, node_color='skyblue', arrowstyle='->', arrowsize=20, width=2)
nx.draw(G, pos, with_labels=True)
# 添加节点和边的标签（可选，因为with_labels=True已经显示了节点标签）
# 你可以使用nx.draw_networkx_edge_labels来添加边标签
edge_labels = {(u, v): d['type'] for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title('Control Flow Graph')
plt.show()
'''


def link_cfg(cur_func, caller, path, vis_graph):
    for called_func in callprivate[cur_func]:
        # print(called_func)
        block = list(called_func.keys())[0]
        dest_func = list(called_func.values())[0]
        # print(f"block = {block}")
        # print(f"dest_func = {dest_func}")
        if dest_func not in caller.nodes:
            callee = nx.read_gpickle(os.path.join(path, f'{dest_func}.gpickle'))
            caller = nx.union(caller, callee)
            caller.add_edge(block, dest_func, type='CALLPRIVATE', graph=caller)
        else:
            caller.add_edge(block, dest_func, type='CALLPRIVATE', graph=caller)
        if dest_func not in vis_graph and len(callprivate[dest_func]) != 0:
            vis_graph.append(dest_func)
            caller = link_cfg(dest_func, caller, path, vis_graph)
    return caller


def is_hex(s):
    # 十六进制正则表达式
    hex_pattern = re.compile(r'^0x[0-9a-fA-F]+$')
    return bool(hex_pattern.match(s))

#########################################################
#                  构建每个函数的CFG                      #
########################################################

def get_func_cfg(path):  # 传入路径参数
    global function_selector, callprivate, calledprivate, vis_block

    vis_block = []
    callprivate = {}
    calledprivate = []
    func2cfg = {}
    function_selector = True

    if os.path.exists(os.path.join(path, 'contract.tac')):

        _, blocks = get_info(os.path.join(path, 'contract.tac'))
        func_entry = pd.read_csv(os.path.join(path, 'IRFunctionEntry.csv'), names=['entry'])['entry'].values  # IRFunctionEntry文件的地址
        # func_entry = [item for key in func_entry for item in key]
        # print(func_entry)
        # print(func_entry)

        for entry in func_entry:
            # if entry == '0x0':  # 0x0 是函数选择器，不需要构建
            #     continue
            callprivate[entry] = []
            # print(entry)
            if entry in vis_block:
                # print(f"{entry}'s cfg has been created!")
                continue
            else:
                G = nx.DiGraph()
                start_block = entry
                block = blocks[start_block]
                vis_block.append(start_block)
                G.add_node(start_block, label=block.block_number, attr=block.instructions)
                construct_cfg(entry, blocks, start_block, G, parent=start_block)
                # 将每个函数的CFG存到func2cfg中
                func2cfg[start_block] = G
                nx.write_gpickle(G, f'{os.path.join(path, start_block+".gpickle")}')
                # print(f"{entry} is OK!")
            # 将每个函数可视化
            function_selector = False

        # 将每个通过CALLPRIVATE调用的函数连接起来
        for func in callprivate.keys():
            caller_g = nx.read_gpickle(os.path.join(path, f'{func}.gpickle'))
            # print(f"{func} before\t: {caller_g.nodes}")
            vis_graph = []
            vis_graph.append(func)
            caller_g = link_cfg(func, caller_g, path, vis_graph)
            nx.write_gpickle(caller_g, f'{path}/{func}_full.gpickle')
            '''
            dot = Digraph('G', filename=f'.{path}/cfg_{func}.gv')
            dot.attr(style='filled', color='lightgrey', rankdir='TB', overlap='scale', splines='polyline', ratio='fill')
            dot.attr('node', shape='rectangle', style='filled', fillcolor='grey',
                     gradientangle='360', label='n9:360', fontcolor='black')
            dot.node_attr.update(style='filled', color='white')
            for node in caller_g.nodes:
                instruction = ''
                dot.node(node, label=node)
    
            for _from, _to in caller_g.edges:
                # print(f'{_from}->{_to}')
                dot.edge(_from, _to, color='blue')
    
            dot.render(filename=f'{path}/cfg_{func}.gv', view=True)
            dot.view()
            '''
        # for file in os.listdir(path):
        #     if file.endswith('.gpickle'):
        #         prefix = file.split('.')[0]
        #         if is_hex(prefix):
        #             os.remove(os.path.join(path, file))
        # print(list(set(calledprivate)))
        # return func2cfg, callprivate, list(set(calledprivate)), _, blocks
        return list(set(calledprivate)), blocks
    else:
        pass
        # shutil.rmtree(path)


# get_func_cfg('../data/Test2/Test2')
# get_func_cfg('./data/YCBToken/TokenERC20')
# get_func_cfg('./data/cross_contract_reentrancy/MoonToken')
# get_func_cfg('./data/ACOToken/ACOToken')
# get_func_cfg("../testdata\\0x0000eee4cbada79cf378e011fbd8b377fedd04a5\\DecoEscrow")
