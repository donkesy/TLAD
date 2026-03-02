import networkx as nx
from web3 import Web3
import numpy as np
import pandas as pd
from graphviz import Digraph
from utils.tac2graph import get_func_cfg, get_info
from collections import defaultdict
import sys
import re
import os
import hashlib
import subprocess

# global self.self.variables
variables = {}
memory = {}
# memory的 key 是'memory_偏移量'
storage = {}
# storage的 key 是'storage_偏移量'
calldata = {}
# self.data_dependency的 key 是变量名， 不是具体的数值
data_dependency = {}
# blocks_ = {}
alchemy_url = 'API KEY'  # 替换为你的Alchemy API密钥
# 匹配括号前的变量名
before_parenthesis = r'\s*(\w+)'
# 匹配括号内的变量值
inside_parenthesis = r'\(([^)]*)\)'
all_one = 0xffffffffffffffffffffffffffffffffffffffff
# 在执行当前块之前执行的是哪个块
before_block = ''
prev = []
# 保存每个合约的内存数据
contract_memory = {}
# 保存每个合约的存储数据
contract_storage = {}
returnprivate_map = defaultdict(lambda: list())
return_map = defaultdict(lambda: list())
block_dependency = defaultdict(lambda: {})
del_edge = defaultdict()
calledbreak = False
returnblock = False
returndata = ''
privatereturn_value = []
CALL_block = None
is_called = 0  # 是否被调用，如果被调用才执行RETURNPRIVATE
pre_returndatasize = ''
jumpi_list = []
vis_node = []
caller_func = ''
caller_block = ''
bytecode_ = ''


def get_balance(address):
    # Alchemy API URL（用自己的Alchemy项目API密钥）
    global alchemy_url

    web3 = Web3(Web3.HTTPProvider(alchemy_url))

    try:
        # 获取账户余额，单位为wei
        balance_wei = web3.eth.get_balance(address)
        # print(f"账户余额: {balance_wei} wei")
        # 将wei转换为ether（1 ether = 10 ** 18 wei）
        # balance_ether = web3.from_wei(balance_wei, 'ether')
        # print(f"账户余额: {balance_ether} ETH")
        return hex(balance_wei)
    except Exception as e:
        # print(f"发生错误：{e}")
        return e


def get_codesize(address):
    global alchemy_url

    web3 = Web3(Web3.HTTPProvider(alchemy_url))

    try:
        # 获取合约代码
        contract_code = web3.eth.get_code(address)
        # 计算字节码的大小（字节码是一个十六进制字符串，所以需要除以2）
        code_size = len(contract_code) // 2
        # print(f"合约代码大小: {code_size} 字节")
        return hex(code_size)
    except Exception as e:
        # print(f"发生错误：{e}")
        return e


def get_bytecode(address):
    global alchemy_url

    web3 = Web3(Web3.HTTPProvider(alchemy_url))

    try:
        bytecode_hex = web3.eth.get_code(address).hex()
        return bytecode_hex.replace('0x', '')
    except Exception as e:
        # print(f"发生错误：{e}")
        return f"error: {e}"


def match_value(value, pattern):
    ans = re.search(pattern, value)
    if ans is None:
        return value
    else:
        return ans.group(1)


def read_bytecode(path):
    with open(path, 'r', encoding='utf-8') as f:
        bytecode_ = f.readline()
    return bytecode_


def is_parenthesis(vname):
    if '(' in vname:
        vname = match_value(vname, before_parenthesis)
    return vname


def parse_arithmetic_operand(res, left_, right_):
    res = is_parenthesis(res)
    left_ = is_parenthesis(left_)
    right_ = is_parenthesis(right_)
    return res, left_, right_


def is_hex(s):
    # 十六进制正则表达式
    hex_pattern = re.compile(r'^0x[0-9a-fA-F]+$')
    return bool(hex_pattern.match(s))


def getcodehash(address):
    if address.startswith('0x'):
        address = address[2:]

    bytecode = get_bytecode(address)
    keccak_hash = Web3.keccak(text=bytecode).hex()
    return keccak_hash


def update_data(datapath, address_value, bytecodebyaddress, ):
    '''
    :param datapath: ./data/xxx
    :param address_value:
    :param bytecodebyaddress:
    :return:
    '''
    os.mkdir(address_value)
    with open(f"{datapath}/{address_value}/{address_value}.hex", 'w', encoding='utf-8') as f:
        f.write(bytecodebyaddress)
    cwd = '../gigahorse-toolchain'
    # 记得修改字节码路径
    command = ['../gigahorse.py', '-C', 'clients/visualizeout.py', f'{datapath}/{address_value}/{address_value}.hex']
    preproc_process = subprocess.run(command, cwd=cwd, universal_newlines=True, capture_output=True)
    if preproc_process.returncode == 0:
        dest_path = f"{datapath}/{address_value}"
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        # print(dest_path)
        command1 = ['cp', f'.temp/{address_value}/out/contract.tac', f'{dest_path}']
        preproc_process = subprocess.run(command1, cwd=cwd, universal_newlines=True, capture_output=True)
        command2 = ['cp', f'.temp/{address_value}/out/IRFunctionEntry.csv', f'{dest_path}']
        preproc_process = subprocess.run(command2, cwd=cwd, universal_newlines=True, capture_output=True)
        command2 = ['cp', f'.temp/{address_value}/out/PublicFunction.csv', f'{dest_path}']
        preproc_process = subprocess.run(command2, cwd=cwd, universal_newlines=True, capture_output=True)


def dataflow_analysis(blocks_, operation, path, cur_func, cur_block, g):  # 通过操作码指令,构建数据流与数据依赖关系
    global variables
    global memory, del_edge
    global storage, returnprivate_map, is_called, vis_node, CALL_block
    global data_dependency, caller_block, caller_func, privatereturn_value
    global bytecode_, calledbreak, returnblock, returndata, pre_returndatasize
    operation = operation.replace(',', '')
    op = operation.split(' ')
    # print(operation)
    # print(op)

    if op[-1] == 'CONST':  ## 定义常量
        # print(op[1])
        value_ = match_value(op[1], inside_parenthesis)  # 匹配括号中的值
        key_ = match_value(op[1], before_parenthesis)  # 匹配括号前的变量
        variables[key_] = str(value_)
        data_dependency[key_] = f'{key_}(CONST)'
        # print(f"{key_} = {value_}")
    elif 'MSTORE' in op or 'SSTORE' in op:  ## 存储操作
        offset_ = op[2]
        offset_ = is_parenthesis(offset_)
        value_ = op[3]
        value_ = is_parenthesis(value_)
        # print(f'offset = {offset_}, value = {value_}')
        if variables[offset_] == 'Unknown':
            return
        if 'MSTORE' in op:
            res = 'memory_' + variables[offset_]
            memory[res] = str(variables[value_])
            data_dependency[res] = data_dependency[value_]
        else:
            res = 'storage_' + variables[offset_]
            storage[res] = str(variables[value_])
            data_dependency[res] = data_dependency[value_]
        # print(f"{offset_}, {value_}")
        # print(f'{res} = {data_dependency[res]}')
    elif 'MLOAD' in op or 'SLOAD' in op:  ## 加载操作, SLOAD 有时后面给出的偏移量的位置之前没有存储操作
        res = op[1]
        vname = op[-1]
        vname = is_parenthesis(vname)
        if 'MLOAD' in op:
            if vname not in variables:
                if '(' in res:
                    temp = res
                    res = is_parenthesis(res)
                    variables[res] = match_value(temp, inside_parenthesis)
                else:
                    variables[res] = 'Unknown'
                data_dependency[res] = f'{res}(the unknown value from memory)'
            else:
                memory_offset = 'memory_' + variables[vname]
                if memory_offset not in memory.keys():
                    if '(' in res:
                        temp = res
                        res = is_parenthesis(res)
                        variables[res] = match_value(temp, inside_parenthesis)
                    else:
                        variables[res] = 'Unknown'
                    data_dependency[res] = f'{res}(the unknown value from memory)'
                else:
                    variables[res] = memory[memory_offset]
                    data_dependency[res] = data_dependency[memory_offset]
            # print(f"self.variables[{res}] = {self.variables[res]}")
        else:
            if vname not in variables:
                variables[res] = 'Unknown'
                data_dependency[res] = '(the unknown value from storage)'
            else:
                if variables[vname] in storage.keys():
                    storage_offset = 'storage_' + variables[vname]
                    variables[res] = storage[storage_offset]
                    data_dependency[res] = data_dependency[storage_offset]
                else:
                    variables[res] = 'Unknown'
                    data_dependency[res] = f'{res}(the unknown value from storage)'
    elif 'JUMPI' in op:  ## 块依赖
        jumpi_condition_ = op[-1]
        jumpi_condition_ = is_parenthesis(jumpi_condition_)
        jumpi_condition_value = variables[jumpi_condition_]
        jumpi_list.append(jumpi_condition_value)
        del_edge = {}
        dest_ = op[-2]
        dest_ = is_parenthesis(dest_)
        dest_block = variables[dest_]
        succs = blocks_[cur_block].succ
        # for succ in succs:
        #     if dest_block in succ or dest_block == succ:
        #         block_dependency[cur_block][dest_block] = f'{data_dependency[jumpi_condition_]}'
        if '0x1' in jumpi_condition_value:
            # print(f"******************{cur_block}")
            for succ in succs:
                if dest_block not in succ:
                    del_edge[cur_block] = succ
        elif '0x0' in jumpi_condition_value:
            # print(f"******************{cur_block}")
            for succ in succs:
                if dest_block in succ or dest_block == succ:
                    del_edge[cur_block] = succ
    elif 'GAS' in op:
        res = op[1]
        res = is_parenthesis(res)
        variables[res] = 'Unknown'
        data_dependency[res] = f'{res}(the gas remaining after this instruction has been executed)'
    elif 'TIMESTAMP' in op:
        res = op[1]
        res = is_parenthesis(res)
        variables[res] = 'UnKnown'
        data_dependency[res] = f'{res}(unix timestamp of the current block)'
    elif 'COINBASE' in op:
        res = op[1]
        res = is_parenthesis(res)
        variables[res] = 'Unknown'
        data_dependency[res] = f"{res}(miner's 20-byte address)"
    elif 'NUMBER' in op:
        res = op[1]
        res = is_parenthesis(res)
        variables[res] = 'Unknown'
        data_dependency[res] = f"{res}(current block number)"
    elif 'CHAINID' in op:
        res = op[1]
        res = is_parenthesis(res)
        variables[res] = 'Unkonwn'
        data_dependency[res] = f"{res}(chain id of the network)"
    elif 'SELFBALANCE' in op:
        res = op[1]
        res = is_parenthesis(res)
        variables[res] = 'Unknown'
        data_dependency[res] = f"{res}(balance of current account in wei)"
    elif 'GASLIMIT' in op:
        res = op[1]
        res = is_parenthesis(res)
        variables[res] = 'Unkonwn'
        data_dependency[res] = f'{res}(the maximum amount of gas to consume on a transaction)'
    # Environmental Information
    elif 'ADDRESS' in op:
        res = op[1]
        res = is_parenthesis(res)
        variables[res] = 'UnKnown'
        data_dependency[res] = f'{res}(the address of the current account)'
    elif 'ORIGIN' in op:
        res = op[1]
        res = is_parenthesis(res)
        variables[res] = 'UnKnown'
        data_dependency[res] = f'{res}(the address of the sender of the transaction)'
    elif 'CALLER' in op:
        res = op[1]
        res = is_parenthesis(res)
        variables[res] = 'UnKnown'
        data_dependency[res] = f'{res}(the address of the caller account)'
    elif 'CALLVALUE' in op:
        res = op[1]
        res = is_parenthesis(res)
        variables[res] = 'Unknown'
        data_dependency[res] = f'{res}(the value of the current call in wei)'
    elif 'CALLDATALOAD' in op:
        res = op[1]
        res = is_parenthesis(res)
        vname = op[-1]
        vname = is_parenthesis(vname)
        variables[res] = 'Unknown'
        data_dependency[res] = f'{res}(in calldata at offset {variables[vname]})'
    elif 'CALLDATASIZE' in op:
        res = op[1]
        res = is_parenthesis(res)
        variables[res] = 'Unknown'
        data_dependency[res] = f'{res}(byte size of the calldata)'
    elif 'CALLDATACOPY' in op:
        dest_offset_ = op[-3]
        dest_offset_ = is_parenthesis(dest_offset_)
        dest_offset_value = variables[dest_offset_]
        offset_ = op[-2]
        offset_ = is_parenthesis(offset_)
        size_ = op[-1]
        size_ = is_parenthesis(size_)
        if dest_offset_value == 'Unknown':
            return
        memory_offset = 'memory_' + dest_offset_value
        memory[memory_offset] = 'Unknown'
        data_dependency[memory_offset] = f'{memory_offset}(in calldata with an offset {variables[offset_]} ' \
                                         f'and a size of {variables[size_]} bytes)'
    elif 'CODESIZE' in op:
        res = op[1]
        res = is_parenthesis(res)
        variables[res] = str(hex(len(bytecode_)))
        data_dependency[res] = f'{res}(byte size of the code)'
    elif 'CODECOPY' in op:
        dest_offset_ = op[-3]
        dest_offset_ = is_parenthesis(dest_offset_)
        dest_offset_value = variables[dest_offset_]
        start_ = op[-2]
        start_ = is_parenthesis(start_)
        start_ = eval(variables[start_])
        offset_ = op[-1]
        offset_ = is_parenthesis(offset_)
        offset_ = eval(variables[offset_])
        res = bytecode_[start_:start_ + offset_]
        if dest_offset_value == 'Unknown':
            return
        memory_offset = 'memory_' + dest_offset_value
        memory[memory_offset] = str('0x' + res)
        data_dependency[memory_offset] = f'{memory_offset}(code whose starting address is {variables[start_]} ' \
                                         f'and offset is {variables[offset_]} running in current environment)'
    elif 'EXTCODESIZE' in op:
        res = op[1]
        res = is_parenthesis(res)
        v_address = op[-1]
        v_address = is_parenthesis(v_address)
        address_ = variables[v_address]
        if is_hex(address_) is False:
            variables[res] = 'Unknown'
        else:
            if '0x' not in address_:
                address_ = '0x' + address_
            variables[res] = str(get_codesize(address_))
        data_dependency[res] = f'{res}(byte size of the code from {address_})'
    elif 'EXTCODECOPY' in op:
        v_address = op[-4]
        v_address = is_parenthesis(v_address)
        address_ = variables[v_address]
        dest_offset = op[-3]
        dest_offset = is_parenthesis(dest_offset)
        start_ = op[-2]
        start_ = is_parenthesis(start_)
        start_ = eval(variables[start_])
        offset_ = op[-1]
        offset_ = is_parenthesis(offset_)
        offset_ = eval(variables[offset_])
        if variables[dest_offset] == 'Unknown':
            return
        memory_offset = 'memory_' + variables[dest_offset]
        if is_hex(address_):
            if '0x' not in address_:
                address_ = '0x' + address_
            bytecode_from_address = get_bytecode(address_)
            res = bytecode_from_address[start_:start_ + offset_]
            memory[memory_offset] = str('0x' + res)
            data_dependency[memory_offset] = f'{memory_offset}(code whose starting address is {variables[start_]} ' \
                                             f'and offset is {variables[offset_]} running in {variables[address_]} environment)'
        else:
            memory[memory_offset] = 'Unknown'
            data_dependency[memory_offset] = f'{memory_offset}(code running in another environment)'
    elif 'EXTCODEHASH' in op:
        res = op[1]
        address_ = op[-1]
        if '(' in address_:
            address_value = match_value(address_, inside_parenthesis)
        else:
            address_value = variables[address_]
        if '(' in res:
            res_value = match_value(res, inside_parenthesis)
            res = is_parenthesis(res)
            variables[res] = res_value
        else:
            if address_value == 'Unkonwn':
                variables[res] = 'Unknown'
            else:
                # if is_hex(address_value):
                exthash = getcodehash(address_value)
                variables[res] = exthash
        data_dependency[res] = f"{res}(the hash of the account code with address {address_value})"
    elif 'BALANCE' in op:
        vname_operand = op[-1]
        vname_operand = is_parenthesis(vname_operand)
        res = op[1]
        res = is_parenthesis(res)
        address_ = variables[vname_operand]
        if is_hex(address_):
            if '0x' not in address_:
                address_ = '0x' + address_
            ether = get_balance(address_)
            variables[res] = str(ether)
        else:
            variables[res] = 'Unknown'
        data_dependency[res] = f'{res}(the balance of ' + data_dependency[vname_operand] + ')'
    # elif 'STOP' in op:
    # Arithmetic Ops
    elif 'ADD' in op:
        res, left_, right_ = parse_arithmetic_operand(op[1], op[-2], op[-1])
        res, left_, right_ = op[1], op[-2], op[-1]
        res = is_parenthesis(res)
        if '(' in left_:
            left_value = match_value(left_, inside_parenthesis)
            left_ = is_parenthesis(left_)
        else:
            left_value = variables[left_]
        if '(' in right_:
            right_value = match_value(right_, inside_parenthesis)
            right_ = is_parenthesis(right_)
        else:
            right_value = variables[right_]
        if is_hex(left_value) is False or is_hex(right_value) is False:
            # print("*******************************************************")
            # print(f'left_value = {left_value}, right_value = {right_value}')
            # print(f'{res} = {"Unknown"}')
            if '(' in op[1]:
                v = match_value(op[1], inside_parenthesis)
                variables[res] = v
            else:
                variables[res] = 'Unknown'
        else:
            res_add = hex(int(left_value, 16) + int(right_value, 16))
            variables[res] = str(res_add)
            # print("*******************************************************")
            # print(f'{res} = {self.variables[res]}')
        data_dependency[res] = f'{res}(ADD {data_dependency[left_]}, {data_dependency[right_]})'
    elif 'MUL' in op:
        res, left_, right_ = parse_arithmetic_operand(op[1], op[-2], op[-1])
        res, left_, right_ = op[1], op[-2], op[-1]
        res = is_parenthesis(res)
        if '(' in left_:
            left_value = match_value(left_, inside_parenthesis)
            left_ = is_parenthesis(left_)
        else:
            left_value = variables[left_]
        if '(' in right_:
            right_value = match_value(right_, inside_parenthesis)
            right_ = is_parenthesis(right_)
        else:
            right_value = variables[right_]
        if is_hex(left_value) is False or is_hex(right_value) is False:
            if '(' in op[1]:
                v = match_value(op[1], inside_parenthesis)
                variables[res] = v
            else:
                variables[res] = 'Unknown'
        else:
            res_mul = hex(int(left_value, 16) * int(right_value, 16))
            variables[res] = str(res_mul)
        data_dependency[res] = f'{res}(MUL {data_dependency[left_]}, {data_dependency[right_]})'
    elif 'SUB' in op:
        res, left_, right_ = parse_arithmetic_operand(op[1], op[-2], op[-1])
        res, left_, right_ = op[1], op[-2], op[-1]
        res = is_parenthesis(res)
        if '(' in left_:
            left_value = match_value(left_, inside_parenthesis)
            left_ = is_parenthesis(left_)
        else:
            left_value = variables[left_]
        if '(' in right_:
            right_value = match_value(right_, inside_parenthesis)
            right_ = is_parenthesis(right_)
        else:
            right_value = variables[right_]
        if is_hex(left_value) is False or is_hex(right_value) is False:
            # print("*******************************************************")
            # print(f'left_value = {left_value}, right_value = {right_value}')
            if '(' in op[1]:
                v = match_value(op[1], inside_parenthesis)
                variables[res] = v
            else:
                variables[res] = 'Unknown'
        else:
            res_sub = hex(int(left_value, 16) - int(right_value, 16))
            # print("*******************************************************")
            # print(f'{res} = {res_sub}')
            variables[res] = str(res_sub)
        data_dependency[res] = f'{res}(SUB {data_dependency[left_]}, {data_dependency[right_]})'
    elif 'DIV' in op:
        res, left_, right_ = parse_arithmetic_operand(op[1], op[-2], op[-1])
        res, left_, right_ = op[1], op[-2], op[-1]
        res = is_parenthesis(res)
        if '(' in left_:
            left_value = match_value(left_, inside_parenthesis)
            left_ = is_parenthesis(left_)
        else:
            left_value = variables[left_]
        if '(' in right_:
            right_value = match_value(right_, inside_parenthesis)
            right_ = is_parenthesis(right_)
        else:
            right_value = variables[right_]
        if is_hex(left_value) is False or is_hex(right_value) is False:
            if '(' in op[1]:
                v = match_value(op[1], inside_parenthesis)
                variables[res] = v
            else:
                variables[res] = 'Unknown'
        else:
            res_div = hex(int(left_value, 16) // int(right_value, 16))
            if eval(right_value) == 0:
                variables[res] = str(hex(0))
            else:
                variables[res] = str(res_div)
        data_dependency[res] = f'{res}(DIV {data_dependency[left_]}, {data_dependency[right_]})'
    elif 'SDIV' in op:
        res, left_, right_ = parse_arithmetic_operand(op[1], op[-2], op[-1])
        res, left_, right_ = op[1], op[-2], op[-1]
        res = is_parenthesis(res)
        if '(' in left_:
            left_value = match_value(left_, inside_parenthesis)
            left_ = is_parenthesis(left_)
        else:
            left_value = variables[left_]
        if '(' in right_:
            right_value = match_value(right_, inside_parenthesis)
            right_ = is_parenthesis(right_)
        else:
            right_value = variables[right_]
        if is_hex(left_value) is False or is_hex(right_value) is False:
            if '(' in op[1]:
                v = match_value(op[1], inside_parenthesis)
                variables[res] = v
            else:
                variables[res] = 'Unknown'
        else:
            res_sdiv = hex(int(left_value, 16) // int(right_value, 16))
            if eval(right_value) == 0:
                variables[res] = str(hex(0))
            else:
                variables[res] = str(res_sdiv)
        data_dependency[res] = f'{res}(SDIV {data_dependency[left_]}, {data_dependency[right_]})'
    elif 'MOD' in op:
        res, left_, right_ = parse_arithmetic_operand(op[1], op[-2], op[-1])
        res, left_, right_ = op[1], op[-2], op[-1]
        res = is_parenthesis(res)
        if '(' in left_:
            left_value = match_value(left_, inside_parenthesis)
            left_ = is_parenthesis(left_)
        else:
            left_value = variables[left_]
        if '(' in right_:
            right_value = match_value(right_, inside_parenthesis)
            right_ = is_parenthesis(right_)
        else:
            right_value = variables[right_]
        if is_hex(left_value) is False or is_hex(right_value) is False:
            if '(' in op[1]:
                v = match_value(op[1], inside_parenthesis)
                variables[res] = v
            else:
                variables[res] = 'Unknown'
        else:
            res_mod = hex(abs(int(variables[left_value], 16)) % abs(int(variables[right_value], 16)))
            if eval(right_value) == 0:
                variables[res] = str(hex(0))
            else:
                variables[res] = str(res_mod)
        data_dependency[res] = f'{res}(MOD {data_dependency[left_]}, {data_dependency[right_]})'
    elif 'SMOD' in op:
        res, left_, right_ = parse_arithmetic_operand(op[1], op[-2], op[-1])
        res, left_, right_ = op[1], op[-2], op[-1]
        res = is_parenthesis(res)
        if '(' in left_:
            left_value = match_value(left_, inside_parenthesis)
            left_ = is_parenthesis(left_)
        else:
            left_value = variables[left_]
        if '(' in right_:
            right_value = match_value(right_, inside_parenthesis)
            right_ = is_parenthesis(right_)
        else:
            right_value = variables[right_]
        if is_hex(left_value) is False or is_hex(right_value) is False:
            if '(' in op[1]:
                v = match_value(op[1], inside_parenthesis)
                variables[res] = v
            else:
                variables[res] = 'Unknown'
        else:
            res_smod = hex(int(variables[left_value], 16) % int(variables[right_value], 16))
            if eval(right_value) == 0:
                variables[res] = str(hex(0))
            else:
                variables[res] = str(res_smod)
        data_dependency[res] = f'{res}(SMOD {data_dependency[left_]}, {data_dependency[right_]})'
    elif 'ADDMOD' in op:
        res = op[1]
        res = is_parenthesis(res)
        left_ = op[-3]
        if '(' in left_:
            left_value = match_value(left_, inside_parenthesis)
            left_ = is_parenthesis(left_)
        else:
            left_value = variables[left_]
        right_ = op[-2]
        if '(' in right_:
            right_value = match_value(right_, inside_parenthesis)
            right_ = is_parenthesis(right_)
        else:
            right_value = variables[right_]
        denominator_ = op[-1]
        if '(' in denominator_:
            denominator_value = match_value(denominator_, inside_parenthesis)
            denominator_ = is_parenthesis(denominator_)
        else:
            denominator_value = variables[denominator_]
        if is_hex(left_value) is False or is_hex(right_value) is False or is_hex(denominator_value) is False:
            if '(' in op[1]:
                v = match_value(op[1], inside_parenthesis)
                variables[res] = v
            else:
                variables[res] = 'Unknown'
        else:
            add_ = int(left_value, 16) + int(right_value, 16)
            if denominator_value == 0:
                res_addmod = hex(0)
            else:
                res_addmod = hex(add_ % int(denominator_value, 16))
            variables[res] = str(res_addmod)
        data_dependency[res] = f'{res}(ADDMOD {data_dependency[left_]}, {data_dependency[right_]}' \
                               f', {data_dependency[denominator_]})'
    elif 'MULMOD' in op:
        res = op[1]
        res = is_parenthesis(res)
        left_ = op[-3]
        if '(' in left_:
            left_value = match_value(left_, inside_parenthesis)
            left_ = is_parenthesis(left_)
        else:
            left_value = variables[left_]
        right_ = op[-2]
        if '(' in right_:
            right_value = match_value(right_, inside_parenthesis)
            right_ = is_parenthesis(right_)
        else:
            right_value = variables[right_]
        denominator_ = op[-1]
        if '(' in denominator_:
            denominator_value = match_value(denominator_, inside_parenthesis)
            denominator_ = is_parenthesis(denominator_)
        else:
            denominator_value = variables[denominator_]
        if is_hex(left_value) is False or is_hex(right_value) is False or is_hex(denominator_value) is False:
            if '(' in op[1]:
                v = match_value(op[1], inside_parenthesis)
                variables[res] = v
            else:
                variables[res] = 'Unknown'
        else:
            mul_ = int(left_value, 16) * int(right_value, 16)
            if denominator_value == 0:
                res_mulmod = hex(0)
            else:
                res_mulmod = hex(mul_ % int(denominator_value, 16))
            variables[res] = str(res_mulmod)
        data_dependency[res] = f'{res}(MULMOD {data_dependency[left_]}, {data_dependency[right_]}' \
                               f', {data_dependency[denominator_]})'
    elif 'EXP' in op:
        res, base_, exp_ = parse_arithmetic_operand(op[1], op[-2], op[-1])
        res, base_, exp_ = op[1], op[-2], op[-1]
        res = is_parenthesis(res)
        if '(' in base_:
            base_value = match_value(base_, inside_parenthesis)
            base_ = is_parenthesis(base_)
        else:
            base_value = variables[base_]
        if '(' in exp_:
            exp_value = match_value(exp_, inside_parenthesis)
            exp_ = is_parenthesis(exp_)
        else:
            exp_value = variables[exp_]
        if is_hex(base_value) is False or is_hex(exp_value) is False:
            if '(' in op[1]:
                v = match_value(op[1], inside_parenthesis)
                variables[res] = v
            else:
                variables[res] = 'Unknown'
        else:
            res_exp = hex(int(base_value, 16) ** int(exp_value, 16))
            variables[res] = str(res_exp)
        data_dependency[res] = f'{res}(EXP {data_dependency[base_]}, {data_dependency[exp_]})'
    # elif 'SIGNEXTEND' in op:
    # Comparison and Bitwise Logic
    elif 'LT' in op or 'SLT' in op:
        # res, left_, right_ = parse_arithmetic_operand(op[1], op[-2], op[-1])
        res, left_, right_ = op[1], op[-2], op[-1]
        res = is_parenthesis(res)
        if '(' in left_:
            left_value = match_value(left_, inside_parenthesis)
            left_ = is_parenthesis(left_)
        else:
            left_value = variables[left_]
        if '(' in right_:
            right_value = match_value(right_, inside_parenthesis)
            right_ = is_parenthesis(right_)
        else:
            right_value = variables[right_]
        if is_hex(left_value) is False or is_hex(right_value) is False:
            variables[res] = 'Unknown'
        else:
            res_lt = (int(left_value, 16) < int(right_value, 16))
            variables[res] = str(hex(res_lt))
        data_dependency[res] = f'{res}(LT {data_dependency[left_]}, {data_dependency[right_]})'
    elif 'GT' in op or 'SGT' in op:
        # print(f"cur func={cur_func}")
        # res, left_, right_ = parse_arithmetic_operand(op[1], op[-2], op[-1])
        res, left_, right_ = op[1], op[-2], op[-1]
        res = is_parenthesis(res)
        if '(' in left_:
            left_value = match_value(left_, inside_parenthesis)
            left_ = is_parenthesis(left_)
        else:
            left_value = variables[left_]
        if '(' in right_:
            right_value = match_value(right_, inside_parenthesis)
            right_ = is_parenthesis(right_)
        else:
            right_value = variables[right_]
        if is_hex(left_value) is False or is_hex(right_value) is False:
            variables[res] = 'Unknown'
        else:
            res_gt = (int(left_value, 16) > int(right_value, 16))
            variables[res] = str(hex(res_gt))
        data_dependency[res] = f'{res}(GT {data_dependency[left_]}, {data_dependency[right_]})'
    elif 'EQ' in op:
        # res, left_, right_ = parse_arithmetic_operand(op[1], op[-2], op[-1])
        res, left_, right_ = op[1], op[-2], op[-1]
        res = is_parenthesis(res)
        if '(' in left_:
            left_value = match_value(left_, inside_parenthesis)
            left_ = is_parenthesis(left_)
        else:
            left_value = variables[left_]
        if '(' in right_:
            right_value = match_value(right_, inside_parenthesis)
            right_ = is_parenthesis(right_)
        else:
            right_value = variables[right_]
        if is_hex(left_value) is False or is_hex(right_value) is False:
            variables[res] = 'Unknown'
        else:
            res_eq = hex(int(left_value, 16) == int(right_value, 16))
            variables[res] = str(res_eq)
        data_dependency[res] = f'{res}(EQ {data_dependency[left_]}, {data_dependency[right_]})'
    elif 'ISZERO' in op:
        res = op[1]
        param_ = op[-1]
        res = is_parenthesis(res)
        if '(' in param_:
            param_value = match_value(param_, inside_parenthesis)
            param_ = is_parenthesis(param_)
        else:
            param_ = is_parenthesis(param_)
            param_value = variables[param_]
        if param_value == 'Unknown' or is_hex(param_value) is False:
            variables[res] = 'Unknown'
        else:
            if eval(param_value) == 0:
                variables[res] = str(hex(1))
            else:
                variables[res] = str(hex(0))
        data_dependency[res] = f'{res}(ISZERO {data_dependency[param_]})'
    elif 'AND' in op:
        # res, left_, right_ = parse_arithmetic_operand(op[1], op[-2], op[-1])
        res, left_, right_ = op[1], op[-2], op[-1]
        res = is_parenthesis(res)
        if '(' in left_:
            left_value = match_value(left_, inside_parenthesis)
            left_ = is_parenthesis(left_)
        else:
            left_value = variables[left_]
        if '(' in right_:
            right_value = match_value(right_, inside_parenthesis)
            right_ = is_parenthesis(right_)
        else:
            right_value = variables[right_]
        if is_hex(left_value) is False or is_hex(right_value) is False:
            variables[res] = 'Unknown'
        else:
            res_and = hex(int(left_value, 16) & int(right_value, 16))
            variables[res] = str(res_and)
        data_dependency[res] = f'{res}(AND {data_dependency[left_]}, {data_dependency[right_]})'
    elif 'OR' in op:
        res, left_, right_ = parse_arithmetic_operand(op[1], op[-2], op[-1])
        res, left_, right_ = op[1], op[-2], op[-1]
        res = is_parenthesis(res)
        if '(' in left_:
            left_value = match_value(left_, inside_parenthesis)
            left_ = is_parenthesis(left_)
        else:
            left_value = variables[left_]
        if '(' in right_:
            right_value = match_value(right_, inside_parenthesis)
            right_ = is_parenthesis(right_)
        else:
            right_value = variables[right_]
        if is_hex(left_value) is False or is_hex(right_value) is False:
            variables[res] = 'Unknown'
        else:
            res_or = hex(int(left_value, 16) | int(right_value, 16))
            variables[res] = str(res_or)
        data_dependency[res] = f'{res}(OR {data_dependency[left_]}, {data_dependency[right_]})'
    elif 'XOR' in op:
        res, left_, right_ = parse_arithmetic_operand(op[1], op[-2], op[-1])
        res, left_, right_ = op[1], op[-2], op[-1]
        res = is_parenthesis(res)
        if '(' in left_:
            left_value = match_value(left_, inside_parenthesis)
            left_ = is_parenthesis(left_)
        else:
            left_value = variables[left_]
        if '(' in right_:
            right_value = match_value(right_, inside_parenthesis)
            right_ = is_parenthesis(right_)
        else:
            right_value = variables[right_]
        if is_hex(left_value) is False or is_hex(right_value) is False:
            variables[res] = 'Unknown'
        else:
            res_xor = hex(int(left_value, 16) | int(right_value, 16))
            variables[res] = str(res_xor)
        data_dependency[res] = f'{res}(XOR {data_dependency[left_]}, {data_dependency[right_]})'
    elif 'NOT' in op:
        res = op[1]
        res = is_parenthesis(res)
        param_ = op[-1]
        if '(' in param_:
            param_value = match_value(param_, inside_parenthesis)
            param_ = is_parenthesis(param_)
        else:
            param_ = is_parenthesis(param_)
            param_value = variables[param_]
        if is_hex(param_value) is False:
            variables[res] = 'Unknown'
        else:
            res_not = hex(~int(param_value, 16) & all_one)
            variables[res] = str(res_not)
        data_dependency[res] = f'{res}(NOT {data_dependency[param_]})'
    elif 'BYTE' in op:
        res, left_, right_ = op[1], op[-2], op[-1]
        res = is_parenthesis(res)
        if '(' in left_:
            left_value = match_value(left_, inside_parenthesis)
            left_ = is_parenthesis(left_)
        else:
            left_value = variables[left_]
        if '(' in right_:
            right_value = match_value(right_, inside_parenthesis)
            right_ = is_parenthesis(right_)
        else:
            right_value = variables[right_]
        if is_hex(left_value) is False or is_hex(right_value) is False:
            variables[res] = 'Unknown'
        else:
            if len(right_value.replace('0x', '')) != 64:
                right_value = right_value.replace('0x', '').zfill(64)
            res_byte = right_value[eval(left_value)*2:eval(left_value)*2+2]
            variables[res] = '0x' + str(res_byte)
        data_dependency[res] = f'{res}(BYTE {data_dependency[left_]}, {data_dependency[right_]})'
    elif 'SHL' in op:
        res, shift_, value_ = parse_arithmetic_operand(op[1], op[-2], op[-1])
        res, shift_, value_ = op[1], op[-2], op[-1]
        res = is_parenthesis(res)
        if '(' in shift_:
            shift_value = match_value(shift_, inside_parenthesis)
            shift_ = is_parenthesis(shift_)
        else:
            shift_value = variables[shift_]
        if '(' in value_:
            value_value = match_value(value_, inside_parenthesis)
            value_ = is_parenthesis(value_)
        else:
            value_value = variables[value_]
        if is_hex(shift_value) is False or is_hex(value_value) is False:
            variables[res] = 'Unknown'
        else:
            if int(shift_value, 16) > 255:
                variables[res] = 0
            else:
                res_shl = str(hex(int(value_value, 16) << int(shift_value, 16)))
                if len(res_shl) > 66:
                    res_shl = res_shl[-64:]
                    res_shl = '0x' + res_shl
                variables[res] = res_shl
        data_dependency[res] = f'{res}(SHL {data_dependency[shift_]}, {data_dependency[value_]})'
    elif 'SHR' in op or 'SAR' in op:
        res, shift_, value_ = parse_arithmetic_operand(op[1], op[-2], op[-1])
        res, shift_, value_ = op[1], op[-2], op[-1]
        res = is_parenthesis(res)
        if '(' in shift_:
            shift_value = match_value(shift_, inside_parenthesis)
            shift_ = is_parenthesis(shift_)
        else:
            shift_value = variables[shift_]
        if '(' in value_:
            value_value = match_value(value_, inside_parenthesis)
            value_ = is_parenthesis(value_)
        else:
            value_value = variables[value_]
        if is_hex(shift_value) is False or is_hex(value_value) is False:
            variables[res] = 'Unknown'
        else:
            if int(shift_value, 16) > 255:
                variables[res] = 0
            else:
                res_shr = str(hex(int(value_value, 16) >> int(shift_value, 16)))
                variables[res] = res_shr
        if 'SHR' in op:
            data_dependency[res] = f'{res}(SHR {data_dependency[shift_]}, {data_dependency[value_]})'
        else:
            data_dependency[res] = f'{res}(SAR {data_dependency[shift_]}, {data_dependency[value_]})'
        # print(f'{res} = {self.variables[res]}')
    elif 'SHA3' in op:
        res, left_, right_ = parse_arithmetic_operand(op[1], op[-2], op[-1])
        res, left_, right_ = op[1], op[-2], op[-1]
        res = is_parenthesis(res)
        if '(' in left_:
            left_value = match_value(left_, inside_parenthesis)
            left_ = is_parenthesis(left_)
        else:
            left_value = variables[left_]
        if '(' in right_:
            right_value = match_value(right_, inside_parenthesis)
            right_ = is_parenthesis(right_)
        else:
            right_value = variables[right_]
        if is_hex(left_value) and is_hex(right_value):
            res_num = ''
            unknown = False
            for i in range(int(left_value, 16), int(right_value, 16), 0x20):
                offset_ = hex(i)
                memory_offset = 'memory_' + offset_
                # print(f"memory's offset = {memory_offset}")
                if memory_offset not in memory.keys():
                    unknown = True
                    variables[res] = 'Unknown'
                    break
                else:
                    if memory[memory_offset] == 'Unknown':
                        unknown = True
                        variables[res] = 'Unknown'
                        break
                    else:
                        res_num += memory[memory_offset].replace('0x', '').zfill(64)
            if unknown is False:
                res_num = res_num.replace('0x', '')
                if len(res_num) % 2 == 1:
                    res_num = '0' + res_num
                res_byte = bytes.fromhex(res_num)
                sha3_hash = hashlib.sha3_256()
                sha3_hash.update(res_byte)
                res_hash = sha3_hash.hexdigest()
                variables[res] = str('0x' + res_hash)
        else:
            variables[res] = 'Unknown'
        data_dependency[res] = f'{res}(SHA3 {data_dependency[left_]}, {data_dependency[right_]})'
    # Logging
    elif 'LOG0' in op:
        loc_ = op[0].replace(':', '')
        offset_ = op[-2]
        offset_ = is_parenthesis(offset_)
        size_ = op[-1]
        data_dependency[
            loc_ + 'LOG0'] = f'(LOG0 memory_offset:{data_dependency[offset_]}, size:{data_dependency[size_]})'
    elif 'LOG1' in op:
        loc_ = op[0].replace(':', '')
        offset_ = op[-3]
        offset_ = is_parenthesis(offset_)
        size_ = op[-2]
        size_ = is_parenthesis(size_)
        topic1_ = op[-1]
        topic1_ = is_parenthesis(topic1_)
        data_dependency[
            loc_ + 'LOG1'] = f'(LOG1 memory_offset:{data_dependency[offset_]}, size:{data_dependency[size_]}, ' \
                             f'topic1:{data_dependency[topic1_]})'
    elif 'LOG2' in op:
        loc_ = op[0].replace(':', '')
        offset_ = op[-4]
        offset_ = is_parenthesis(offset_)
        size_ = op[-3]
        size_ = is_parenthesis(size_)
        topic1_ = op[-2]
        topic1_ = is_parenthesis(topic1_)
        topic2_ = op[-1]
        topic2_ = is_parenthesis(topic2_)
        data_dependency[
            loc_ + 'LOG2'] = f'(LOG2 memory_offset:{data_dependency[offset_]}, size:{data_dependency[size_]}, ' \
                             f'topic1:{data_dependency[topic1_]}, topic2:{data_dependency[topic2_]})'
    elif 'LOG3' in op:
        loc_ = op[0].replace(':', '')
        offset_ = op[-5]
        offset_ = is_parenthesis(offset_)
        size_ = op[-4]
        size_ = is_parenthesis(size_)
        topic1_ = op[-3]
        topic1_ = is_parenthesis(topic1_)
        topic2_ = op[-2]
        topic2_ = is_parenthesis(topic2_)
        topic3_ = op[-1]
        topic3_ = is_parenthesis(topic3_)
        data_dependency[
            loc_ + 'LOG3'] = f'(LOG3 memory_offset:{data_dependency[offset_]}, size:{data_dependency[size_]}, ' \
                             f'topic1:{data_dependency[topic1_]}, topic2:{data_dependency[topic2_]},' \
                             f'topic3:{data_dependency[topic3_]})'
    elif 'CREATE' in op:
        res, value_, offset_, size_ = op[1], op[-3], op[-2], op[-1]
        variables[res] = 'UnKnown'
        data_dependency[res] = f"the address of contract created by CREATE opcode"
    # PHI 有问题
    elif 'PHI' in op:  # 有问题
        res = op[1]
        res = is_parenthesis(res)
        params_length = len(op) - 4
        params = []
        # prev2var = {}
        for i in range(params_length):
            loc = params_length * (-1) + i
            params.append(op[loc])
            params[i] = is_parenthesis(params[i])

        variables[res] = 'Unknown'
        data_dependency[res] = f'{res}(PHI '
        for param in params:
            if param not in data_dependency.keys():
                data_dependency[res] += 'Unknown'
            else:
                data_dependency[res] += f'{data_dependency[param]}'
            if param != params[-1]:
                data_dependency[res] += ','
        data_dependency[res] += ')'
        # print(data_dependency[res])
    elif 'RETURNDATACOPY' in op:
        destOffset_ = op[-3]
        destOffset_ = is_parenthesis(destOffset_)
        destOffset_value = variables[destOffset_]
        offset_ = op[-2]
        offset_ = is_parenthesis(offset_)
        offset_value = variables[offset_]
        size_ = op[-1]
        size_ = is_parenthesis(size_)
        size_value = variables[size_]
        if destOffset_value == 'Unknown':
            return
        # print(returndata)
        if is_hex(returndata):
            memory['memory_' + destOffset_value] = '0x' + returndata[offset_value + 2:offset_value + size_value + 2]
        else:
            memory['memory_' + destOffset_value] = 'Unknown'
        data_dependency['memory_' + destOffset_value] = f'memory_{destOffset_value}(copy output data from the previous call to memory)'
    elif 'RETURNDATASIZE' in op:
        res = op[1]
        res = is_parenthesis(res)
        variables[res] = pre_returndatasize
        data_dependency[res] = f'{res}(size of output data from the previous call from the current environment)'
    # 构建返回边
    elif 'RETURN' in op and op[1] == 'RETURN' and returnblock:  # size 没有用到
        returnblock = False
        calledbreak = True
        offset_ = op[-2]
        offset_ = is_parenthesis(offset_)
        size_ = op[-1]
        size_ = is_parenthesis(size_)
        offset_value = variables[offset_]
        memory_offset = f'memory_{offset_value}'
        if offset_value != 'Unknown':
            if memory_offset not in memory.keys():
                returndata = 'Unknown'
            else:
                returndata = memory[memory_offset]
        else:
            returndata = 'Unknown'
            # print("current nodes:")
            # print(g.nodes)
        # g.add_edge(cur_block, f'{CALL_block}_return', type='RETURN')
        return_map[cur_block].append(f"{CALL_block}_return")
    elif 'RETURNPRIVATE' in op and is_called > 0:
        '''
        0xf1e0xb6d: RETURNPRIVATE vb6darg3(返回到哪个块)
        0xf3a: RETURNPRIVATE vf1farg2(返回到哪个块), vf24(返回值)

        '''
        is_called -= 1
        len_return = len(op)
        if len_return == 3:
            dest_func_ = op[-1]
            dest_func_ = is_parenthesis(dest_func_)
            dest_func_value = variables[dest_func_]
        else:
            dest_func_ = op[-2]
            dest_func_ = is_parenthesis(dest_func_)
            dest_func_value = variables[dest_func_]
            for i in range(3, len(op)):
                privatereturn_ = op[i]
                privatereturn_ = is_parenthesis(privatereturn_)
                privatereturn_value.append(privatereturn_)
        # print(f"returnprivate -> {dest_func_}:{dest_func_value}")
        # print(f"privatereturn_value:{privatereturn_value}")

        succ = blocks_[cur_block].succ
        for succ_value in succ:
            if dest_func_value in succ_value or dest_func_value == succ_value:
                dest_func_value = succ_value
                break
        returnprivate_map[cur_block].append(dest_func_value)
    elif 'CALLPRIVATE' in op and op[1] == 'CALLPRIVATE' and cur_func != '0x0':
        # 无返回值
        is_called += 1
        caller_func = cur_block
        caller_block = cur_block

        # 如果CALLPRIVATE位于函数选择器中，不会执行调用
        # 无返回值的CALLPRIVATE
        dest_func_ = op[2]
        dest_func_ = match_value(dest_func_, inside_parenthesis)
        dest_func_value = dest_func_.replace('0x', '')
        idx = 0
        params = []
        for i in range(3, len(op)):
            new_param = 'v' + dest_func_value + 'arg' + str(idx)
            idx += 1
            param = op[i]
            param = is_parenthesis(param)
            params.append(new_param)
            variables[new_param] = variables[param]
            data_dependency[new_param] = data_dependency[param]
        succ = blocks_[cur_block].succ
        for succ_value in succ:
            # print(succ_value)
            if variables[params[-1]] in succ_value or variables[params[-1]] == succ_value:
                variables[params[-1]] = succ_value
                break
        # print(f"self.variables[{params[-1]}] = {self.variables[params[-1]]}")
        private_g = nx.read_gpickle(os.path.join(path, f'{dest_func_}_full.gpickle'))

        for node in private_g.nodes:
            instructions = private_g.nodes[node]['attr']
            for line in instructions:
                dataflow_analysis(blocks_, line, path, dest_func_, node, g)

    # elif 'CALLPRIVATE' in op and len(op) > 3 and op[3] == 'CALLPRIVATE' and cur_func != '0x0':
    elif 'CALLPRIVATE' in op and '=' in op and cur_func != '0x0':
        # 有返回值的CALLPRIVATE
        # op[3] == 'CALLPRIVATE'，表示有返回值
        # 不知道 returnprivate 的参数是什么意思，导致res的值也不确定
        is_called += 1
        caller_func = cur_block
        caller_block = cur_block

        equal_loc = op.index('=')
        res = []
        for i in range(1, equal_loc):
            temp = op[i]
            temp = is_parenthesis(temp)
            res.append(temp)

        # res = op[1]
        # res = self.is_parenthesis(res)
        dest_func_ = op[equal_loc+2]
        dest_func_ = match_value(dest_func_, inside_parenthesis)  # 0x1ab
        dest_func_value = dest_func_.replace('0x', '')  # 1ab
        # params_len = len(op) - 5
        params = []
        idx = 0
        for i in range(5, len(op)):
            new_param = 'v' + dest_func_value + 'arg' + str(idx)
            idx += 1
            param = op[i]
            param = is_parenthesis(param)
            params.append(new_param)
            variables[new_param] = variables[param]
            data_dependency[new_param] = data_dependency[param]

        succ = blocks_[cur_block].succ
        for succ_value in succ:
            # print(succ_value)
            if variables[params[-1]] in succ_value or variables[params[-1]] == succ_value:
                variables[params[-1]] = succ_value
                break

        # print(f"self.variables[{params[-1]}] = {self.variables[params[-1]]}")
        # 对private函数进行数据流/依赖分析
        private_g = nx.read_gpickle(os.path.join(path, f'{dest_func_}.gpickle'))
        for node in private_g.nodes:
            instructions = private_g.nodes[node]['attr']
            for line in instructions:
                dataflow_analysis(blocks_, line, path, dest_func_, node, g)

        for i in range(len(res)):
            pop_value = privatereturn_value.pop()
            variables[res[i]] = variables[pop_value]
            data_dependency[res[i]] = data_dependency[pop_value]

        # dependency = f'(CALLPRIVATE function:0x{dest_func_value}, '
        # for i in range(len(params)):
        #     dependency += f'arg{i}:{data_dependency[params[i]]}'
        #     if i != len(params) - 1:
        #         dependency += ', '
        #     else:
        #         dependency += ')'
        # self.data_dependency[res] = dependency
    # 构建跨合约调用边, 也不一定是跨合约，注意！！
    # elif 'CALLPRIVATE' in op and op[]
    elif 'DELEGETACALL' in op and op[3] == 'DELEGATECALL':
        CALL_block = cur_block
        res = op[1]
        res = is_parenthesis(res)
        retSize_ = op[-1]
        retSize_ = is_parenthesis(retSize_)
        retOffset_ = op[-2]
        retOffset_ = is_parenthesis(retOffset_)
        argsSize_ = op[-3]
        argsSize_ = is_parenthesis(argsSize_)
        argsSize_value = variables[argsSize_]
        # print(f'argsSize_ = {argsSize_}, argsSize_value = {argsSize_value}')
        argsOffset_ = op[-4]
        argsOffset_ = is_parenthesis(argsOffset_)
        argsOffset_value = variables[argsOffset_]
        address_ = op[-5]
        address_ = is_parenthesis(address_)
        address_value = variables[address_]
        gas_ = op[-7]
        gas_ = is_parenthesis(gas_)
        if argsSize_value == 'Unknown':
            argsSize_value = 8
        else:
            argsSize_value = int(argsSize_value, 16) * 2

        if argsOffset_value == 'Unknown':
            variables[res] = str(hex(0))
            data_dependency[res] = f'{res}(whether the function call was successful)'
            returndata = 'Unknown'
            return

        in_memory_offset = "memory_" + argsOffset_value
        if in_memory_offset not in memory.keys():
            variables[res] = str(hex(0))
            data_dependency[res] = f'{res}(whether the function call was successful)'
            returndata = 'Unknown'
            if variables[retOffset_] == 'Unkonwn':
                return
            out_memory_offset = 'memory_' + variables[retOffset_]
            pre_returndatasize = variables[retSize_]
            memory[out_memory_offset] = 'Unknown'
            data_dependency[out_memory_offset] = f'{out_memory_offset}(return data from calling Unknown function)'
            return
        if is_hex(memory["memory_" + argsOffset_value]) is False:
            variables[res] = str(hex(0))
        calledfunc_sign = memory["memory_" + argsOffset_value][0:argsSize_value + 2]
        # print(f'argsOffset = {argsOffset_value}, argsSize = {argsSize_value}')
        # print(f'function\'s signature = {memory["memory_" + argsOffset_value][0:argsSize_value + 2]}')
        # 判断哪个合约中存在这个函数签名，然后对这个合约进行数据流和数据依赖分析，一直遇到被调用的函数中的RETURN指令，结束运行
        # 注意修改datapath
        # datapath = './data/Test2'
        # 返回上一级目录
        datapath = os.path.dirname(path)
        flag = False
        # 匹配签名相同的函数所在的合约
        if is_hex(address_value):
            bytecodeByaddress = get_bytecode(address_value)
            if "error" not in bytecodeByaddress:
                update_data(datapath, address_value, bytecodeByaddress)
        for root, dirs, files in os.walk(datapath):
            for folderdir in dirs:
                new_path = os.path.join(datapath, folderdir)
                if new_path == path:
                    continue
                public_function = pd.read_csv(os.path.join(new_path, 'PublicFunction.csv'), header=None,
                                              names=['entry', 'signature'], sep='\t')
                signature2entry = public_function.set_index('signature')['entry'].to_dict()
                # print(signature2entry)
                if calledfunc_sign in signature2entry.keys():
                    # print(f'sign = {calledfunc_sign}, block = {signature2entry[calledfunc_sign]}')
                    _, blocks = get_info(os.path.join(new_path, 'contract.tac'))
                    # 保存递归前的数据，递归结束后再恢复
                    prev_variables = variables
                    prev_data_dependency = data_dependency
                    prev_memory = memory
                    prev_storage = storage
                    prev_vis_node = vis_node

                    # self.data_dependency = {}
                    # self.self.variables = {}
                    # storage = {}
                    # memory = {}

                    # calledblock 是被调用函数块的标号
                    calledblock = signature2entry[calledfunc_sign]
                    # traverse_block(blocks, new_path, calledblock)
                    traverse_func(blocks, new_path, calledblock, calledblock)
                    callee_G = nx.read_gpickle(os.path.join(new_path, f'{calledblock}_full.gpickle'))
                    g.add_nodes_from(callee_G.nodes)
                    g.add_edges_from(callee_G.edges)
                    g.add_edge(cur_block, calledblock, type='DELEGATECALL', graph=g)
                    # print(os.path.join(path, f'{cur_func}_full.gpickle'))
                    variables = prev_variables
                    data_dependency = prev_data_dependency
                    memory = prev_memory
                    storage = prev_storage
                    vis_node = prev_vis_node
                    flag = True
                    variables[res] = str(hex(1))
                    pre_returndatasize = variables[retSize_]
                    break
            if flag:
                break
        if flag is False:
            variables[res] = str(hex(0))
        data_dependency[res] = f'{res}(whether the function staticcall was successful)'
        if variables[retOffset_] == 'Unknown':
            return
        out_memory_offset = 'memory_' + variables[retOffset_]
        memory[out_memory_offset] = returndata
        data_dependency[
            out_memory_offset] = f'{out_memory_offset}(DELEGATECALL gas:{variables[gas_]}, address:{variables[address_]}, ' \
                                 f'argsOffset:{variables[argsOffset_]}, argsSize:{variables[argsSize_]}, ' \
                                 f'retOffset:{variables[retOffset_]}, retSize:{variables[retSize_]})'
    elif 'STATICCALL' in op and op[3] == 'STATICCALL':
        CALL_block = cur_block
        res = op[1]
        res = is_parenthesis(res)
        retSize_ = op[-1]
        retSize_ = is_parenthesis(retSize_)
        retOffset_ = op[-2]
        retOffset_ = is_parenthesis(retOffset_)
        argsSize_ = op[-3]
        argsSize_ = is_parenthesis(argsSize_)
        argsSize_value = variables[argsSize_]
        # print(f'argsSize_ = {argsSize_}, argsSize_value = {argsSize_value}')
        argsOffset_ = op[-4]
        argsOffset_ = is_parenthesis(argsOffset_)
        argsOffset_value = variables[argsOffset_]
        address_ = op[-5]
        address_ = is_parenthesis(address_)
        address_value = variables[address_]
        gas_ = op[-7]
        gas_ = is_parenthesis(gas_)
        if argsSize_value == 'Unknown':
            argsSize_value = 8
        else:
            argsSize_value = int(argsSize_value, 16) * 2

        if argsOffset_value == 'Unknown':
            variables[res] = str(hex(0))
            data_dependency[res] = f'{res}(whether the function call was successful)'
            returndata = 'Unknown'
            return

        in_memory_offset = "memory_" + argsOffset_value
        if in_memory_offset not in memory.keys():
            variables[res] = str(hex(0))
            data_dependency[res] = f'{res}(whether the function call was successful)'
            returndata = 'Unknown'
            if variables[retOffset_] == 'Unkonwn':
                return
            out_memory_offset = 'memory_' + variables[retOffset_]
            pre_returndatasize = variables[retSize_]
            memory[out_memory_offset] = 'Unknown'
            data_dependency[out_memory_offset] = f'{out_memory_offset}(return data from calling Unknown function)'
            return
        if is_hex(memory["memory_" + argsOffset_value]) is False:
            variables[res] = str(hex(0))
        calledfunc_sign = memory["memory_" + argsOffset_value][0:argsSize_value + 2]
        # print(f'argsOffset = {argsOffset_value}, argsSize = {argsSize_value}')
        # print(f'function\'s signature = {memory["memory_" + argsOffset_value][0:argsSize_value + 2]}')
        # 判断哪个合约中存在这个函数签名，然后对这个合约进行数据流和数据依赖分析，一直遇到被调用的函数中的RETURN指令，结束运行
        # 注意修改datapath
        # datapath = './data/Test2'
        # 返回上一级目录
        datapath = os.path.dirname(path)
        flag = False
        # 匹配签名相同的函数所在的合约
        if is_hex(address_value):
            bytecodeByaddress = get_bytecode(address_value)
            if "error" not in bytecodeByaddress:
                update_data(datapath, address_value, bytecodeByaddress)
        for root, dirs, files in os.walk(datapath):
            for folderdir in dirs:
                new_path = os.path.join(datapath, folderdir)
                if new_path == path:
                    continue
                public_function = pd.read_csv(os.path.join(new_path, 'PublicFunction.csv'), header=None,
                                              names=['entry', 'signature'], sep='\t')
                signature2entry = public_function.set_index('signature')['entry'].to_dict()
                # print(signature2entry)
                if calledfunc_sign in signature2entry.keys():
                    # print(f'sign = {calledfunc_sign}, block = {signature2entry[calledfunc_sign]}')
                    _, blocks = get_info(os.path.join(new_path, 'contract.tac'))
                    # 保存递归前的数据，递归结束后再恢复
                    prev_variables = variables
                    prev_data_dependency = data_dependency
                    prev_memory = memory
                    prev_storage = storage
                    prev_vis_node = vis_node

                    data_dependency = {}
                    variables = {}
                    storage = {}
                    memory = {}

                    # calledblock 是被调用函数块的标号
                    calledblock = signature2entry[calledfunc_sign]
                    # traverse_block(blocks, new_path, calledblock)
                    traverse_func(blocks, new_path, calledblock, calledblock)
                    callee_G = nx.read_gpickle(os.path.join(new_path, f'{calledblock}_full.gpickle'))
                    g.add_nodes_from(callee_G.nodes)
                    g.add_edges_from(callee_G.edges)
                    g.add_edge(cur_block, calledblock, type='STATICCALL', graph=g)
                    # print(os.path.join(path, f'{cur_func}_full.gpickle'))
                    variables = prev_variables
                    data_dependency = prev_data_dependency
                    memory = prev_memory
                    storage = prev_storage
                    vis_node = prev_vis_node
                    flag = True
                    variables[res] = str(hex(1))
                    pre_returndatasize = variables[retSize_]
                    break
            if flag:
                break
        if flag is False:
            variables[res] = str(hex(0))
        data_dependency[res] = f'{res}(whether the function staticcall was successful)'
        if variables[retOffset_] == 'Unknown':
            return
        out_memory_offset = 'memory_' + variables[retOffset_]
        memory[out_memory_offset] = returndata
        data_dependency[
            out_memory_offset] = f'{out_memory_offset}(STATICCALL gas:{variables[gas_]}, address:{variables[address_]}, ' \
                                 f'argsOffset:{variables[argsOffset_]}, argsSize:{variables[argsSize_]}, ' \
                                 f'retOffset:{variables[retOffset_]}, retSize:{variables[retSize_]})'
        # print(data_dependency[out_memory_offset])
    elif 'CALL' in op and 'CALL' == op[3]:  # size 没有用到
        CALL_block = cur_block
        res = op[1]
        res = is_parenthesis(res)
        retSize_ = op[-1]
        retSize_ = is_parenthesis(retSize_)
        retOffset_ = op[-2]
        retOffset_ = is_parenthesis(retOffset_)
        argsSize_ = op[-3]
        argsSize_ = is_parenthesis(argsSize_)
        argsSize_value = variables[argsSize_]
        # print(f'argsSize_ = {argsSize_}, argsSize_value = {argsSize_value}')
        argsOffset_ = op[-4]
        argsOffset_ = is_parenthesis(argsOffset_)
        value_ = op[-5]
        value_ = is_parenthesis(value_)
        address_ = op[-6]
        address_ = is_parenthesis(address_)
        address_value = variables[address_]
        gas_ = op[-7]
        gas_ = is_parenthesis(gas_)
        argsOffset_value = variables[argsOffset_]
        # print(argsOffset_value)
        # if is_hex(argsOffset_value) is False:
        # print(memory.keys())
        # print(memory.values())
        if argsSize_value == 'Unknown':
            argsSize_value = 8
        else:
            argsSize_value = int(argsSize_value, 16) * 2

        if argsOffset_value == 'Unknown':
            variables[res] = str(hex(0))
            data_dependency[res] = f'{res}(whether the function call was successful)'
            returndata = 'Unknown'
            return

        in_memory_offset = "memory_" + argsOffset_value
        if in_memory_offset not in memory.keys():
            variables[res] = str(hex(0))
            data_dependency[res] = f'{res}(whether the function call was successful)'
            returndata = 'Unknown'
            if variables[retOffset_] == 'Unkonwn':
                return
            out_memory_offset = 'memory_' + variables[retOffset_]
            pre_returndatasize = variables[retSize_]
            memory[out_memory_offset] = 'Unknown'
            data_dependency[out_memory_offset] = f'{out_memory_offset}(return data from calling Unknown function)'
            return
        if is_hex(memory["memory_" + argsOffset_value]) is False:
            variables[res] = str(hex(0))
        calledfunc_sign = memory["memory_" + argsOffset_value][0:argsSize_value + 2]
        # print(f'argsOffset = {argsOffset_value}, argsSize = {argsSize_value}')
        # print(f'function\'s signature = {memory["memory_" + argsOffset_value][0:argsSize_value + 2]}')
        # 判断哪个合约中存在这个函数签名，然后对这个合约进行数据流和数据依赖分析，一直遇到被调用的函数中的RETURN指令，结束运行
        # 注意修改datapath
        # datapath = './data/Test2'
        # 返回上一级目录
        datapath = os.path.dirname(path)
        flag = False
        # 匹配签名相同的函数所在的合约
        if is_hex(address_value):
            bytecodeByaddress = get_bytecode(address_value)
            if "error" not in bytecodeByaddress:
                update_data(datapath, address_value, bytecodeByaddress)
        for root, dirs, files in os.walk(datapath):
            for folderdir in dirs:
                new_path = os.path.join(datapath, folderdir)
                if new_path == path:
                    continue
                public_function = pd.read_csv(os.path.join(new_path, 'PublicFunction.csv'), header=None,
                                              names=['entry', 'signature'], sep='\t')
                signature2entry = public_function.set_index('signature')['entry'].to_dict()
                # print(signature2entry)
                if calledfunc_sign in signature2entry.keys():
                    # print(f'sign = {calledfunc_sign}, block = {signature2entry[calledfunc_sign]}')
                    _, blocks = get_info(os.path.join(new_path, 'contract.tac'))
                    # 保存递归前的数据，递归结束后再恢复
                    prev_variables = variables
                    prev_data_dependency = data_dependency
                    prev_memory = memory
                    prev_storage = storage
                    prev_vis_node = vis_node

                    data_dependency = {}
                    variables = {}
                    storage = {}
                    memory = {}

                    # calledblock 是被调用函数块的标号
                    calledblock = signature2entry[calledfunc_sign]
                    # traverse_block(blocks, new_path, calledblock)
                    traverse_func(blocks, new_path, calledblock, calledblock)
                    callee_G = nx.read_gpickle(os.path.join(new_path, f'{calledblock}_full.gpickle'))
                    # caller_G = nx.read_gpickle(os.path.join(path, f'{cur_func}_full.gpickle'))
                    # print(os.path.join(new_path, f'{calledblock}_full.gpickle'))
                    # print(os.path.join(path, f'{cur_func}_full.gpickle'))
                    # G = nx.union(G, callee_G)
                    g.add_nodes_from(callee_G.nodes)
                    g.add_edges_from(callee_G.edges)
                    g.add_edge(cur_block, calledblock, type='CALL', graph=g)

                    node_attrs = {node: callee_G.nodes[node]['attr'] for node in callee_G.nodes}
                    for node, attr in node_attrs.items():
                        # print(node, attr)
                        g.nodes[node]['attr'] = attr

                    g.remove_edge(cur_block, f'{cur_block}_return')
                    # print("*******************************")
                    # print(f"cur block = {cur_block}, calledblock = {calledblock}")
                    # print(g.nodes)
                    # print(os.path.join(path, f'{cur_func}_full.gpickle'))
                    # nx.write_gpickle(g, os.path.join(path, f'{cur_func}_full.gpickle'))
                    ##################################
                    #        删除被调用函数的cfg        #
                    ##################################
                    variables = prev_variables
                    data_dependency = prev_data_dependency
                    memory = prev_memory
                    storage = prev_storage
                    vis_node = prev_vis_node
                    flag = True
                    variables[res] = str(hex(1))
                    pre_returndatasize = variables[retSize_]
                    break
            if flag:
                break
        if flag is False:
            variables[res] = str(hex(0))
        data_dependency[res] = f'{res}(whether the function call was successful)'
        if variables[retOffset_] == 'Unknown':
            return
        out_memory_offset = 'memory_' + variables[retOffset_]
        memory[out_memory_offset] = returndata
        data_dependency[
            out_memory_offset] = f'{out_memory_offset}(CALL gas:{variables[gas_]}, address:{variables[address_]}, value:{variables[value_]}, ' \
                                 f'argsOffset:{variables[argsOffset_]}, argsSize:{variables[argsSize_]}, ' \
                                 f'retOffset:{variables[retOffset_]}, retSize:{variables[retSize_]})'
        # print(data_dependency[out_memory_offset])



def traverse_func(blocks_, graphpath, func, calledblock):
    global before_block, memory, storage, variables, data_dependency, \
        returnblock, calledbreak, vis_node, is_called, returnprivate_map
    # calledbreak 调用是否结束
    # returnblock 只有当前函数被调用时才执行return指令

    # self.data_dependency = {}
    # self.self.variables = {}
    # storage = {}
    # memory = {}

    is_called = 0
    returnprivate_map = defaultdict(lambda: list(set()))

    G = nx.read_gpickle(os.path.join(graphpath, f"{func}_full.gpickle"))
    vis_node = []
    returnblock = False
    G_copy = G.copy()
    for node in G_copy.nodes:
        instructions = G_copy.nodes[node]['attr']
        if node in vis_node:
            continue
        if calledblock == node:
            returnblock = True
        for line in instructions:
            dataflow_analysis(blocks_, line, graphpath, func, node, G)  # func = cur_func, node = cur_block
        if calledbreak is True:
            calledbreak = False
            return
    # print(f"{func}: return private: {returnprivate_map}")
    # G = nx.read_gpickle(os.path.join(graphpath, f"{func}_full.gpickle"))
    # print(G.nodes)
    for u, v_s in returnprivate_map.items():
        v_s = list(set(v_s))
        for v in v_s:
            G.add_edge(u, v, type='RETURNPRIVATE', graph=G)
    # print(f"{func}: return: {return_map}")
    for u, v_s in return_map.items():
        v_s = list(set(v_s))
        for v in v_s:
            G.add_edge(u, v, type='RETURN', graph=G)
    # for u, v in del_edge.items():
        # print(f"u = {u}, v = {v}")
        # G.remove_edge(u, v)
    # print(f"del_edge:{del_edge}")
    nx.write_gpickle(G, os.path.join(graphpath, f"{func}_full.gpickle"))


if __name__ == "__main__":
    # path = '../data/YCBToken/TokenERC20'
    path = '../backup/Test2/Test2'
    # path = '../data/ACOToken/ACOToken'
    # path = '../testdata/0x0000eee4cbada79cf378e011fbd8b377fedd04a5/DecoEscrow'
    # 构建了每个函数的full控制流图，构/建了RETURNPRIVATE边
    calledprivate, blocks_ = get_func_cfg(path)
    print(f"calledprivate = {calledprivate}")
    # sys.exit(0)
    # functions_, blocks_ = get_info(os.path.join(path, 'contract.tac'))
    # print(func2cfg)
    # print(blocks_)
    # calledblock = 'False'
    # traverse_block(blocks_, path, calledblock='False')

    # entry_blocks = pd.read_csv(os.path.join(path, 'IRFunctionEntry.csv'), header=None, names=['entry'], sep='\t')
    entry_blocks = pd.read_csv(os.path.join(path, 'PublicFunction.csv'), header=None, names=['entry', 'signature'], sep='\t')
    entry_blocks = entry_blocks['entry'].values
    traverse_func(blocks_, path, '0x0', calledblock='False')
    # print(entry_blocks)
    # sys.exit(0)
    for block in entry_blocks:

        if block in calledprivate:
            # print(f"block = {block}")
            continue
        print(f"block = {block}")
        cur_func = block
        traverse_func(blocks_, path, cur_func, calledblock='False')

        # with open(f'./data_dependency_{path.split("/")[-1]}.txt', 'w', encoding='utf-8') as f:
        #     for key_ in self.data_dependency.keys():
        #         f.write(f"{data_dependency[key_]}\n")
