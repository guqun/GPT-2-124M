"""12：注解不做运行时校验；CLI 的边界显式验证。"""
import argparse
from typing import Optional


def annotated(value: int) -> int:
    return value


def validate_id(token_id: int, vocab_size: int) -> int:
    if type(token_id) is not int or not 0 <= token_id < vocab_size:
        raise ValueError("token_id 必须是 0 到 vocab_size - 1 之间的整数")
    return token_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-id", type=int, default=220)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    try:
        token_id = validate_id(args.token_id, 50257)
    except ValueError as error:
        parser.error(str(error))
    print("已检查 ID：", token_id)
    print("是否显示全部：", args.all)
    result = annotated("仍然可以传入字符串")
    print("注解没有拦截：", type(result).__name__)
    maybe_id: Optional[int] = None
    print("Optional 的运行时值：", maybe_id)
    try:
        try:
            int("abc")
        except ValueError as error:
            raise RuntimeError("解析配置失败") from error
    except RuntimeError as error:
        print("保留异常链：", type(error).__name__, "<-", type(error.__cause__).__name__)
    assert validate_id(0, 50257) == 0


if __name__ == "__main__":
    main()
