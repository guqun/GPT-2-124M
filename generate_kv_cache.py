"""B：prefill 后每次只送入一个新 Token，逐层复用历史 K/V。"""
from kv_cache_experiment import main

if __name__=='__main__': main(True)
