"""B：在相同主干上替换 SwiGLU，使用近似匹配参数量的中间宽度。"""
from swiglu_training import main

if __name__ == '__main__':
    main('swiglu')
