"""加载第23块自己训练的模型，使用同一套 ModernLM 进行缓存/重算生成。"""
import argparse
from pathlib import Path
from train_modern import ROOT,torch,tiktoken,load_checkpoint,prepare_local_data,resolve_device,generation_check


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'.cache/modern-integrated/default/checkpoint.pt')
    p.add_argument('--prompt',default="Hello, I'm a language model,")
    p.add_argument('--new-tokens',type=int,default=16)
    p.add_argument('--device',choices=['cpu','mps','cuda','auto'],default='cpu')
    args=p.parse_args();torch.set_num_threads(4)
    try:
        device=resolve_device(args.device)
        model,saved=load_checkpoint(args.checkpoint,device)
        prepare_local_data(saved['spec'])  # 核对本地分词资产，禁止隐式下载。
        enc=tiktoken.get_encoding('gpt2')
        ids=torch.tensor([enc.encode_ordinary(args.prompt)],dtype=torch.long,device=device)
        if ids.shape[1]==0: raise ValueError('提示词不能为空')
        ids,check=generation_check(model,ids,args.new_tokens)
        print(enc.decode(ids[0].tolist()));print(check)
    except (OSError,ValueError) as error: p.exit(1,str(error)+'\n')


if __name__=='__main__': main()
