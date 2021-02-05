#! -*- coding: utf-8 -*-
# desc:
import numpy as np

def is_string(s):
    """判断是否是字符串
    """
    return isinstance(s, str)


def convert_to_unicode(text, encoding='utf-8', errors='ignore'):
    """字符串转换为unicode格式（假设输入为utf-8格式）
    """
    
    if isinstance(text, bytes):
        text = text.decode(encoding, errors=errors)
    return text


def to_array(*args):
    """批量转numpy的array
    """
    results = [np.array(a) for a in args]
    if len(args) == 1:
        return results[0]
    else:
        return results