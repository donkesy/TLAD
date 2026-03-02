import os
import subprocess
from utils.get_version import update_version
from loguru import logger


def split_file(inputFileDir, content):
    if not os.path.exists(inputFileDir):
        os.makedirs(inputFileDir)
    content = content.split('\n')
    for i in range(len(content)):
        # print(content[i])
        if content[i][0:6] == '======':
            file_name = content[i].replace("=", "").split(':')[1].strip()
            file_name = file_name.replace('\n', '')
            # print(file_name)
            hex = content[i + 2]
            # print(hex)
            if len(hex) == 0:
                continue
            hexdir = os.path.join(inputFileDir, file_name)
            if not os.path.exists(hexdir):
                os.makedirs(hexdir)
            path = hexdir + "/" + file_name + '.hex'
            # print(path)
            with open(path, "w", encoding='utf-8') as f_w:
                f_w.write(hex)

path = '/home/yuh/MyGit/gigahorse-toolchain/examples'
cwd = '/home/yuh/MyGit/gigahorse-toolchain'
outputpath = '/home/yuh/MyGit/gigahorse-toolchain/output'
'''
inputdata
        |full_contract1
                        |contract1
                                  |contract.tac
                                  |IRFunctionEntry.csv
                                  |PublicFunction.csv
                        |contract2
                                  |contract.tac
                                  |IRFunctionEntry.csv
                                  |PublicFunction.csv
'''
logger.remove()
logger.add("gigahorse_log.txt")

for dirs in os.listdir(path):
    if os.path.isdir(os.path.join(path, dirs)):
        for filename in os.listdir(os.path.join(path, dirs)):
            if filename.endswith('.sol'):
                # 编译生成字节码
                update_version(os.path.join(path, dirs, filename))
                command = ['solc', '--bin-runtime', f'examples/{dirs}/{filename}']
                print(command)
                preproc_process = subprocess.run(command, cwd=cwd, universal_newlines=True, capture_output=True)
                result_path = os.path.dirname(path) + '/' + 'inputdata' + '/' + dirs
                split_file(result_path, preproc_process.stdout) #将每份字节码分配到一个文件中

                # 使用gigahorse反编译字节码
                for filehex in os.listdir(result_path):
                    if filehex.endswith('.hex'):
                        command = ['./gigahorse.py', '-C', 'clients/visualizeout.py', f'inputdata/{dirs}/{filehex}']
                        preproc_process = subprocess.run(command, cwd=cwd, universal_newlines=True, capture_output=True)
                        print(preproc_process)
                        file_without_hex = filehex.replace('.hex', '')
                        if preproc_process.returncode == 0:
                            logger.info(preproc_process.stdout)
                            dest_path = os.path.join(result_path, file_without_hex)
                            if not os.path.exists(dest_path):
                                os.makedirs(dest_path)
                            # print(dest_path)
                            command1 = ['cp', f'.temp/{file_without_hex}/out/contract.tac', f'{dest_path}']
                            preproc_process = subprocess.run(command1, cwd=cwd, universal_newlines=True, capture_output=True)
                            command2 = ['cp', f'.temp/{file_without_hex}/out/IRFunctionEntry.csv', f'{dest_path}']
                            preproc_process = subprocess.run(command2, cwd=cwd, universal_newlines=True, capture_output=True)
                            command2 = ['cp', f'.temp/{file_without_hex}/out/PublicFunction.csv', f'{dest_path}']
                            preproc_process = subprocess.run(command2, cwd=cwd, universal_newlines=True, capture_output=True)
                        else:
                            logger.info(preproc_process.stderr)



