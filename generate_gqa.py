"""第 21 块 B：4 个 Q 头、2 个 KV 头；每两个 Q 共用一组 K/V。"""
try:
    from gqa_experiment import main
except ModuleNotFoundError as error:
    raise SystemExit(f'缺少课程依赖：{error}\n课程解释器：/Users/guqun/micrograd/venv/bin/python')

if __name__ == '__main__':
    main(2)
