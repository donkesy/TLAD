import os
import re
import subprocess
import time
from loguru import logger

pattern_version = r"pragma\s+solidity\s+([^\s;]+);"
range_version = r"pragma\s+solidity\s+(>=\d+\.\d+\.\d+\s*<\s*\d+\.\d+\.\d+);"
patter_middle = r"\.(\d+)\."
patter_tail = r"\.\.(\d+)"
number_pattern = r"\d+\.\d+\.\d+"
# path = '/home/yuh/SmartContract/Test'
# path = '/home/yuh/SmartContract/Slise/All_contract'
# path = 'E:\Dataset\cross-contract\Slise\All_contract'
cur_version = '0.4.26'
cwd = '/home/yuh/MyGit/gigahorse-toolchain'


preproc_command = ['solc-select', 'versions']
preproc_process = subprocess.run(preproc_command, universal_newlines=True, capture_output=True)
installed_version = preproc_process.stdout


logger.add("./logs/install_logs.log")

def create_binary(file):
    _command = ['solc', '--bin-runtime', file]
    _process = subprocess.run(_command, universal_newlines=True, capture_output=True)


def install_version(new_version):
    _command = ['solc-select', 'install', f'{new_version}']
    _process = subprocess.run(_command, universal_newlines=True, capture_output=True)
    print(_process)
    if _process.returncode != 0:
        logger.error(f"Failed to install version {new_version}")
        install_version(new_version)
    return _process.returncode


def use_version(new_version):
    _command = ['solc-select', 'use', f'{new_version}']
    _process = subprocess.run(_command, universal_newlines=True,capture_output=True)
    return _process


def update_sol(filepath):
    update_content = ''
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            update_content = update_content + line
        pattern = r'0\.\d+\.\d+;'

        # 替换后的字符串
        replacement = '0.4.24;'

        # 使用正则表达式进行替换
        updated_content = re.sub(pattern, replacement, update_content)

        # 打印或写入文件
        # print(updated_content)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(updated_content)


def check_version(new_version, installed_version):
    if new_version not in installed_version:
        install_version(new_version)
        use_version(new_version)
    else:
        use_version(new_version)


def update_version(path):
    global cur_version
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace(' ', '')
            if "pragmasolidity" in line:
                # print(line)
                res = re.findall(number_pattern, line)
                print(re.findall(number_pattern, line))
                if len(res) == 1:
                    version = res[0]
                    cur_version_q0 = cur_version.replace('0.', '')
                    version_q0 = version.replace('0.', '')
                    tail_version = version_q0.replace(f'{version_q0[0]}.', '')
                    tail_cur_version = cur_version.replace(f'{cur_version_q0[0]}.', '')
                    if cur_version == version:
                        pass
                        # 直接编译生成字节码
                    elif version_q0[0] == '4' and eval(tail_version) < 24:
                        use_version("0.4.24")
                        update_sol(path)
                    elif '^' in line and cur_version[2] == version[2]:
                        if eval(tail_cur_version) < eval(tail_version):
                            installed_version = subprocess.run(preproc_command, universal_newlines=True, capture_output=True).stdout
                            check_version(version, installed_version)
                            cur_version = version
                    else:
                        installed_version = subprocess.run(preproc_command, universal_newlines=True, capture_output=True).stdout
                        check_version(version, installed_version)
                        cur_version = version
                        # 编译生成字节码
                else:
                    low_version = res[0]
                    high_version = res[1]
                    low_version_q0 = res[0].replace('0.', '')
                    high_version_q0 = res[1].replace('0.', '')
                    cur_version_q0 = cur_version.replace('0.', '')
                    print(f'cur_version:{cur_version_q0}, low_version:{low_version}, high_version:{high_version}')
                    if low_version == cur_version:
                        # 编译生成字节码
                        pass
                    elif cur_version_q0[0] == low_version_q0[0]:
                        tail_cur_version = cur_version_q0.replace(f'{cur_version_q0[0]}.', '')
                        tail_version = low_version_q0.replace(f'{low_version_q0[0]}.', '')
                        if eval(tail_cur_version) < eval(tail_version):
                            installed_version = subprocess.run(preproc_command, universal_newlines=True, capture_output=True).stdout
                            check_version(low_version, installed_version)
                            cur_version = low_version
                    elif cur_version_q0[0] == high_version_q0[0]:
                        tail_cur_version = cur_version.replace(f'{cur_version_q0[0]}.', '')
                        tail_version = high_version_q0.replace(f'{high_version_q0[0]}.', '')
                        if eval(tail_cur_version) > eval(tail_version):
                            installed_version = subprocess.run(preproc_command, universal_newlines=True, capture_output=True).stdout
                            check_version(low_version, installed_version)
                            cur_version = low_version
                    elif eval(cur_version_q0) < eval(low_version_q0) or eval(cur_version_q0) > eval(high_version_q0):
                        cur_version = low_version
                        installed_version = subprocess.run(preproc_command, universal_newlines=True, capture_output=True).stdout
                        check_version(cur_version, installed_version)
                        # 编译生成字节码
                break