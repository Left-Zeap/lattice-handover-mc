"""允许执行 ``python -m continuous_loading``。"""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
