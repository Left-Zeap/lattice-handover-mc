"""``python -m ui`` 入口。

启动代码包在 ``main()`` 与 ``__name__`` 守卫中，保证 Windows 多进程
spawn 时子进程不会重复创建 QApplication。
"""

import sys


def main() -> int:
    from .app import run_application

    return run_application(sys.argv)


if __name__ == "__main__":
    sys.exit(main())
