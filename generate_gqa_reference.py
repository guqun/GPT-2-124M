"""第 21 块 A：4 个 Q 头、4 个 KV 头的 MHA 参照；保留所有旧入口。"""
try:
    from gqa_experiment import main
except ModuleNotFoundError as error:
    raise SystemExit(f'缺少课程依赖：{error}\n课程解释器：/Users/guqun/micrograd/venv/bin/python')

if __name__ == '__main__':
    main(4)
