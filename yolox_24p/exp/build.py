import importlib
import os
import sys

# 读取.py文件中的exp类，并且返回该类
def get_exp_by_file(exp_file):
    try:
        sys.path.append(os.path.dirname(exp_file))
        current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])
        exp = current_exp.Exp()

    except Exception:
        raise ImportError("{} doesn't contains class named 'Exp'".format(exp_file))
    return exp


def get_exp(exp_file):
    """
    get Exp object by file

    Args:
        exp_file (str): file path of experiment.
    """
    # 首先判定输入的exp_file和模型名字至少有一个不是空的
    assert (
        exp_file is not None
    ), "plz provide exp file."
    
    # 如果文件名字不为空，就读取文件
    return get_exp_by_file(exp_file)
    