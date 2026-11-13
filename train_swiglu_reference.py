"""A：保留 RoPE + RMSNorm/no-bias + GELU，独立运行参照组。"""
from swiglu_training import main

if __name__ == '__main__':
    main('gelu')
