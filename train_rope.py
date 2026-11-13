"""本节 B：相同随机起点，移除位置表并在每层 Q/K 使用 RoPE。"""
from rope_training import main

if __name__ == '__main__':
    main('rope')
