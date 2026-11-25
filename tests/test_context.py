"""验证业务输入之间的引用关系及形状，不计算目标输出。"""
import numpy as np
from support.inspect_data import load_bundle

def test_linked_materials():
    _, arrays, r, scenarios = load_bundle()
    shape = arrays['parameter'].shape
    assert len(r['vocabulary']['token_ids']) == shape[0]
    for part in r['parts']:
        assert arrays[part['tensor']].shape == (len(part['row_ids']),shape[1])
        assert all(0<=i<shape[0] for i in part['row_ids'])
    assert arrays['parameter_v2'].shape[0] == len(r['new_vocabulary']['token_ids'])
