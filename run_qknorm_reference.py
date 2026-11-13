"""参照：已有 GQA，不加 QK-Norm；保留所有旧入口。"""
try:
    from qknorm_experiment import main
except ModuleNotFoundError as error:
    raise SystemExit(f'缺少课程依赖：{error}\n课程解释器：/Users/guqun/micrograd/venv/bin/python')
if __name__=='__main__': main(False)
