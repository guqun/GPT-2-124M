"""B 入口：相同 GPT-2 起点，替换 RMSNorm 并移除线性 bias。"""
from architecture_training import main

if __name__ == '__main__':
    main('rmsnorm_no_bias')
