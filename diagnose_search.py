"""诊断各搜索方案的可达性。"""
import sys
import time

def test_bingcnsearch():
    print("=" * 60)
    print("测试 1: bingcnsearch")
    print("=" * 60)
    try:
        import bingcnsearch
        import inspect
        print("模块路径:", bingcnsearch.__file__)
        # 看 search 函数的真实签名
        sig = inspect.signature(bingcnsearch.search)
        print("search 签名:", sig)
        print("模块内函数:", [n for n in dir(bingcnsearch) if not n.startswith("_")])
        print("\n尝试默认调用:")
        try:
            results = bingcnsearch.search("北京旅游景点")
            print(f"返回类型: {type(results)}")
            if isinstance(results, list):
                print(f"返回 {len(results)} 条")
                for r in results[:2]:
                    print(" -", r)
        except Exception as e:
            print(f"默认调用失败: {type(e).__name__}: {e}")
    except ImportError as e:
        print(f"未安装: {e}")
    print()


def test_baidu():
    print("=" * 60)
    print("测试 2: 百度搜索")
    print("=" * 60)
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        print("需要 pip install requests beautifulsoup4 lxml")
        return

    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36"),
        "Accept-Language": "zh-CN,zh;q=0.9",
    }
    try:
        resp = requests.get(
            "https://www.baidu.com/s",
            params={"wd": "北京旅游景点"},
            headers=headers,
            timeout=15,
        )
        print(f"状态码: {resp.status_code}")
        print(f"响应长度: {len(resp.text)} 字符")
        print(f"前 300 字符: {resp.text[:300]!r}")
        print()
        soup = BeautifulSoup(resp.text, "lxml")
        # 多种选择器都试一遍
        for sel in [".result", ".c-container", "h3 a", "[tpl]"]:
            items = soup.select(sel)
            print(f"选择器 {sel!r}: 命中 {len(items)} 个")
        # 标题
        title = soup.find("title")
        if title:
            print(f"页面标题: {title.get_text()}")
    except Exception as e:
        print(f"失败: {type(e).__name__}: {e}")
    print()


def test_sogou():
    print("=" * 60)
    print("测试 3: 搜狗搜索")
    print("=" * 60)
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        return

    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36"),
        "Accept-Language": "zh-CN,zh;q=0.9",
    }
    try:
        resp = requests.get(
            "https://www.sogou.com/web",
            params={"query": "北京旅游景点"},
            headers=headers,
            timeout=15,
        )
        print(f"状态码: {resp.status_code}")
        print(f"响应长度: {len(resp.text)} 字符")
        print(f"前 300 字符: {resp.text[:300]!r}")
        soup = BeautifulSoup(resp.text, "lxml")
        for sel in [".vrwrap", ".rb", "h3 a", ".results .vrwrap"]:
            items = soup.select(sel)
            print(f"选择器 {sel!r}: 命中 {len(items)} 个")
    except Exception as e:
        print(f"失败: {type(e).__name__}: {e}")
    print()


def test_cn_bing():
    print("=" * 60)
    print("测试 4: 必应中国 (cn.bing.com)")
    print("=" * 60)
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        return

    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36"),
        "Accept-Language": "zh-CN,zh;q=0.9",
    }
    try:
        resp = requests.get(
            "https://cn.bing.com/search",
            params={"q": "北京旅游景点"},
            headers=headers,
            timeout=15,
        )
        print(f"状态码: {resp.status_code}")
        print(f"响应长度: {len(resp.text)} 字符")
        print(f"前 300 字符: {resp.text[:300]!r}")
        soup = BeautifulSoup(resp.text, "lxml")
        for sel in ["li.b_algo", ".b_algo", "h2 a"]:
            items = soup.select(sel)
            print(f"选择器 {sel!r}: 命中 {len(items)} 个")
    except Exception as e:
        print(f"失败: {type(e).__name__}: {e}")
    print()


if __name__ == "__main__":
    test_bingcnsearch()
    test_baidu()
    test_sogou()
    test_cn_bing()
    print("=" * 60)
    print("诊断完成")
    print("=" * 60)
