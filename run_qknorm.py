"""改动：GQA + 头内 QK-Norm；缓存核验及副本梯度检查。"""
try:
    from qknorm_experiment import main
except ModuleNotFoundError as error:
    raise SystemExit(f'缺少课程依赖：{error}\n课程解释器：/Users/guqun/micrograd/venv/bin/python')
if __name__=='__main__': main(True)
