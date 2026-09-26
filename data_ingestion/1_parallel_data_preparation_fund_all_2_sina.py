import logging
import time
import pandas as pd
import akshare as ak
import random
from logging.handlers import RotatingFileHandler
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import chardet
import numpy as np
import sys
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime_paths import FUND_CRAWLER_LOG_DIR, FUND_LIST_CSV, QLIB_CHANGE_CSV_DIR, QLIB_PROVIDER_URI, QLIB_SCRIPTS_DIR


def normalize_path(p: str) -> str:
    """Normalize configured paths.

    - Replace a leading literal "$home" with the user's home directory.
    - Convert backslashes to the OS path separator.
    - Normalize the final path.
    """
    if not isinstance(p, str):
        return p
    # Replace Windows backslashes with OS separator
    p = p.replace('\\', os.sep)
    # If user used forward slashes, make them OS-consistent
    p = p.replace('/', os.sep)
    # Expand $home placeholder to the actual user home
    if p.startswith('$home' + os.sep) or p == '$home' or p.startswith('$home'):
        p = p.replace('$home', os.path.expanduser('~'), 1)
    return os.path.normpath(p)

# 初始化日志配置
def setup_logger():
    """配置日志系统（含滚动日志功能）"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # 设置根日志级别
    # 创建格式化器（网页1/网页4推荐格式）
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台处理器（INFO级别）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # 创建日志目录（新增代码）
    log_dir = normalize_path(str(FUND_CRAWLER_LOG_DIR))
    os.makedirs(log_dir, exist_ok=True)

    error_handler = RotatingFileHandler(
        os.path.join(log_dir, 'error.log'),
        maxBytes=2 * 1024 * 1024,
        backupCount=3,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

    # 文件处理器（DEBUG级别，滚动日志）
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, 'fund_crawler.log'),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


logger = setup_logger()


# 东财前复权接口返回的中文列 → 项目内统一的英文列映射
_EM_COLUMN_MAP = {
    '日期': 'date',
    '开盘': 'open',
    '收盘': 'close',
    '最高': 'high',
    '最低': 'low',
    '成交量': 'volume',
}


def _strip_market_prefix(symbol: str) -> str:
    """去掉 sz/sh 前缀，东财接口只接受 6 位纯数字代码。"""
    s = str(symbol).strip().lower()
    if s.startswith(('sh', 'sz')):
        s = s[2:]
    return s


# 进程级缓存：避免 akshare 对每只基金都重抓一次 ETF 代码表（分页爬 800+ 只）。
# 同时用一把锁串行化首次加载，防止 8 个线程同时击穿。
import threading as _threading
from functools import lru_cache as _lru_cache
from akshare.fund import fund_etf_em as _ak_fund_etf_em  # type: ignore

_EM_CODE_MAP_LOCK = _threading.Lock()
_EM_PATCHED = False


def _install_em_code_map_cache() -> None:
    """把 akshare 的 _fund_etf_code_id_map_em 替换为 lru_cache 版本。只需安装一次。"""
    global _EM_PATCHED
    if _EM_PATCHED:
        return
    with _EM_CODE_MAP_LOCK:
        if _EM_PATCHED:
            return
        original = _ak_fund_etf_em._fund_etf_code_id_map_em

        @_lru_cache(maxsize=1)
        def _cached():
            return original()

        _ak_fund_etf_em._fund_etf_code_id_map_em = _cached
        _EM_PATCHED = True
        logger.info("已为 akshare 的 _fund_etf_code_id_map_em 安装 lru_cache（进程级缓存）")


def _fetch_em_qfq(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """调用东财前复权接口，返回标准化为英文列名的 DataFrame。

    用东财的前复权 (qfq) 可以正确处理 ETF 分红/拆分，避免出现隔夜跳空
    的脏数据（例如 sh515880 在 2026-02-02→02-03 出现的 -66% 跳空）。
    """
    _install_em_code_map_cache()
    pure_code = _strip_market_prefix(symbol)
    df = ak.fund_etf_hist_em(
        symbol=pure_code,
        period='daily',
        start_date=str(start_date),
        end_date=str(end_date),
        adjust='qfq',
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=_EM_COLUMN_MAP)
    keep_cols = [c for c in ('date', 'open', 'close', 'high', 'low', 'volume') if c in df.columns]
    return df[keep_cols].copy()


# ---------- baostock 兜底（无复权，仅在东财网络全断时使用）----------
_BS_LOCK = _threading.Lock()
_BS_LOGGED_IN = False


def _baostock_login_once() -> bool:
    """懒加载登录 baostock。线程安全，多次调用只登录一次。"""
    global _BS_LOGGED_IN
    if _BS_LOGGED_IN:
        return True
    with _BS_LOCK:
        if _BS_LOGGED_IN:
            return True
        try:
            import baostock as bs  # noqa: WPS433
            lg = bs.login()
            if str(lg.error_code) != '0':
                logger.warning(f"baostock 登录失败 code={lg.error_code} msg={lg.error_msg}")
                return False
            _BS_LOGGED_IN = True
            logger.info("baostock 登录成功（用于东财失败时的兜底抓取）")
            return True
        except Exception as e:
            logger.warning(f"baostock 登录异常: {e}")
            return False


def _fetch_bs(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """用 baostock 兜底。注意：baostock 对 ETF 复权因子支持不完整，
    返回的实际上接近不复权价，仅用于东财 API 全断时的 best-effort 兜底。
    列名已对齐为项目内英文字段。
    """
    if not _baostock_login_once():
        return pd.DataFrame()
    import baostock as bs  # noqa: WPS433
    pure_code = _strip_market_prefix(symbol)
    prefix = 'sh' if symbol.lower().startswith('sh') else ('sz' if symbol.lower().startswith('sz') else None)
    if prefix is None:
        # 按 6 位代码规则兜底：5 开头→sh，1 开头→sz
        prefix = 'sh' if pure_code.startswith('5') else 'sz'
    bs_code = f'{prefix}.{pure_code}'
    # baostock 日期格式需要 YYYY-MM-DD
    sd = f'{str(start_date)[:4]}-{str(start_date)[4:6]}-{str(start_date)[6:8]}'
    ed = f'{str(end_date)[:4]}-{str(end_date)[4:6]}-{str(end_date)[6:8]}'
    rs = bs.query_history_k_data_plus(
        bs_code,
        'date,open,high,low,close,volume',
        start_date=sd,
        end_date=ed,
        frequency='d',
        adjustflag='2',  # 2=前复权（对 ETF 通常等同不复权）
    )
    if str(rs.error_code) != '0':
        logger.warning(f"baostock 查询 {bs_code} 失败: {rs.error_msg}")
        return pd.DataFrame()
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=rs.fields)
    for c in ('open', 'high', 'low', 'close', 'volume'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df[['date', 'open', 'high', 'low', 'close', 'volume']].dropna(subset=['close']).reset_index(drop=True)


def fetch_fund_data_with_retry(
        symbol: str,
        start_date,
        end_date,
        max_retries: int = 3,
        base_timeout: int = 10,
        backoff_factor: float = 1.5
):
    """
    带重试机制的基金数据抓取（主用东财前复权，失败回退到 sina 未复权）。

    :param symbol: 基金代码（形如 'sh515880' / 'sz159915'）
    :param start_date: YYYYMMDD 字符串
    :param end_date:   YYYYMMDD 字符串
    :param max_retries: 最大重试次数(默认3次)
    :param base_timeout: 基础超时时间(秒)（当前未用于真正的 timeout 控制，仅作日志）
    :param backoff_factor: 退避因子(指数退避)
    """
    current_timeout = base_timeout
    last_exc = None
    # 先尝试东财前复权
    for attempt in range(max_retries):
        try:
            df = _fetch_em_qfq(symbol, start_date, end_date)
            if df is not None and not df.empty:
                return df
            logger.warning(f"基金 {symbol} 东财 qfq 返回空，第{attempt + 1}次尝试")
        except Exception as e:
            last_exc = e
            logger.warning(
                f"基金 {symbol} 东财 qfq 第{attempt + 1}次抓取异常: {str(e)}"
            )
        sleep_time = backoff_factor ** attempt
        time.sleep(sleep_time)
        current_timeout = int(current_timeout * 1.2)

    # 东财全部重试失败，第二级降级：baostock（注意：实测 baostock 对 ETF
    # 复权因子支持不完整，通常返回的实际是不复权价，仅当东财全断时用作数据不中断保障）
    logger.warning(
        f"基金 {symbol} 东财 qfq 抓取多次失败，尝试 baostock 兜底。"
        f"最后一次异常：{last_exc!r}"
    )
    try:
        df_bs = _fetch_bs(symbol, start_date, end_date)
        if df_bs is not None and not df_bs.empty:
            return df_bs
        logger.warning(f"基金 {symbol} baostock 返回空")
    except Exception as e:
        logger.warning(f"基金 {symbol} baostock 兜底异常: {e}")

    # 第三级降级：sina（原始路径）
    logger.error(
        f"基金 {symbol} 东财+baostock 均失败，最后降级到 sina 未复权接口。"
    )
    try:
        return ak.fund_etf_hist_sina(symbol=symbol)
    except Exception as e:
        logger.error(f"基金 {symbol} sina 兜底也失败: {str(e)}", exc_info=True)
        raise


def fetch_and_save_data_parallel(dir_name, start_date, end_date):
    """并行化基金数据抓取"""
    try:
        logger.info("开始获取基金列表")
        codefundsecname = ak.fund_exchange_rank_em()
        ####################
        codefundsecname = codefundsecname
        # 确保基金代码为字符串类型（避免数字类型导致的判断错误）
        codefundsecname['基金代码'] = codefundsecname['基金代码'].astype(str)

        # 1开头的加'sz'，5开头的加'sh'，其他保持原样
        codefundsecname['基金代码'] = np.where(
            codefundsecname['基金代码'].str.startswith('1'),  # 条件1：以1开头
            'sz' + codefundsecname['基金代码'],  # 满足条件1的结果
            np.where(
                codefundsecname['基金代码'].str.startswith('5'),  # 条件2：以5开头
                'sh' + codefundsecname['基金代码'],  # 满足条件2的结果
                codefundsecname['基金代码']  # 都不满足时保持原样
            )
        )
        ##################################
        max_date = codefundsecname['日期'].max()
        logger.debug(f"获取到最新基金数据日期：{max_date}")
        # 保存基金列表
        file_name = normalize_path(str(FUND_LIST_CSV))
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        codefundsecname.to_csv(file_name, index=False)
        logger.info(f"基金列表已保存至：{file_name}")

        total_count = len(codefundsecname)
        success_count = 0
        failed_codes = []
        codes = codefundsecname['基金代码'].tolist()

        # Ensure target directory exists (dir_name may contain $home placeholder)
        dir_name = normalize_path(dir_name)
        os.makedirs(dir_name, exist_ok=True)

        # 创建线程池（网页5推荐方式）
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(
                    process_single_fund,  # 封装的单基金处理函数
                    code=code,
                    idx=idx + 1,
                    total=total_count,
                    dir_name=dir_name,
                    start_date=start_date,
                    end_date=end_date
                ): code for idx, code in enumerate(codes)
            }

            # 进度监控（网页4的完成度跟踪）
            for future in as_completed(futures):
                code = futures[future]
                try:
                    result = future.result()
                    if result:
                        success_count += 1
                        logger.debug(f"基金 {code} 处理完成")
                except Exception as e:
                    logger.error(f"基金 {code} 处理失败: {str(e)}", exc_info=True)
                    failed_codes.append(code)

        # 汇总日志
        logger.info(f"任务完成：成功 {success_count}/{total_count}")
        if failed_codes:
            logger.warning(f"失败基金代码列表：{failed_codes}")

    except Exception as e:
        logger.critical("主程序异常终止", exc_info=True)
        raise


def process_single_fund(code, idx, total, dir_name, start_date, end_date):
    """单支基金处理流程（封装为独立函数）"""
    try:
        logger.debug(f"开始处理基金 {code} ({idx}/{total})")

        # 数据抓取（带重试机制）
        fund_df = fetch_fund_data_with_retry(
            code, start_date, end_date,
            max_retries=5,
            base_timeout=15
        )

        if fund_df.empty:
            logger.warning(f"基金 {code} 无有效数据")
            return False

        # 数据清洗
        fund_df = fund_df.rename(columns={
            'date': 'date', 'open': 'open', 'close': 'close',
            'high': 'high', 'low': 'low', 'volume': 'volume'
        })
        fund_df["date"] = pd.to_datetime(fund_df['date']).dt.strftime('%Y-%m-%d')
        fund_df["code"] = code

        # 保存数据
        file_path = os.path.join(dir_name, f"{code}.csv")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        fund_df.to_csv(file_path, index=False)
        logger.debug(f"基金 {code} 数据已保存至 {file_path}")
        return True

    except Exception as e:
        raise  # 异常抛给上层处理
    finally:
        # 添加随机延迟（网页3的反爬建议）
        delay = random.randint(1, 3)
        time.sleep(delay)


if __name__ == '__main__':
    try:
        logger.info("程序启动")
        start_date = '20050101'
        end_date = '20301202'
        csv_path = normalize_path(str(QLIB_CHANGE_CSV_DIR))

        fetch_and_save_data_parallel(csv_path, start_date, end_date)
        logger.info("程序正常退出")

        # 步骤2: 调用QLib数据转换脚本（全量替换数据）
        if QLIB_SCRIPTS_DIR is None:
            raise SystemExit("qlib_scripts_dir is unset. Set paths.qlib_scripts_dir in the local json.")
        qlib_scripts_path = normalize_path(str(QLIB_SCRIPTS_DIR))
        qlib_dir = normalize_path(str(QLIB_PROVIDER_URI))

        # 检查路径是否存在（参考[1,5](@ref)）
        for path in [qlib_scripts_path, csv_path, qlib_dir]:
            if not os.path.exists(path):
                logger.warning(f"路径不存在：{path}")

        # 在当前 Python 环境中运行 QLib 脚本（使用 sys.executable）并先检查依赖
        python_exe = sys.executable

        # 预检查关键依赖 fire，避免子进程报错难以定位
        if importlib.util.find_spec('fire') is None:
            logger.error(
                "缺少依赖模块 'fire'。请在运行脚本的 Python 环境中安装它，例如：\n"
                f"{python_exe} -m pip install fire"
            )
            raise RuntimeError("缺少依赖模块 'fire'，已在日志中给出安装命令。")

        # 构造命令参数列表（使用列表形式避免注入风险）
        command = [
            python_exe,
            f'{qlib_scripts_path}/dump_bin.py',
            'dump_all',
            '--data_path', csv_path,
            '--qlib_dir', qlib_dir,
            '--symbol_field_name', 'code',
            '--date_field_name', 'date',
            '--include_fields', 'open,high,low,close,volume',
        ]

        try:
            # 执行命令并捕获原始字节（避免编码问题[3,6](@ref)）
            result = subprocess.run(command, check=True, capture_output=True)
            logger.info("命令执行成功！输出：")
        except subprocess.CalledProcessError as e:
            # 同样处理错误输出（参考[5,6](@ref)）
            detected_encoding = chardet.detect(e.stderr)['encoding']
            decoded_stderr = e.stderr.decode(detected_encoding)
            logger.error(f"命令执行失败，错误码：{e.returncode}")
            logger.error("错误信息：")
            logger.error(decoded_stderr)
        except KeyboardInterrupt:
            logger.warning("程序被用户中断（Ctrl+C）")
            # 清理操作（参考[5,6](@ref)）
            logger.info("清理完成，程序退出")
        finally:
            # 确保所有清理操作在此执行（解决SyntaxError[5,6](@ref)）
            logger.info("执行最终清理")

    except Exception as e:
        logger.exception("未捕获的全局异常")  # 自动记录堆栈跟踪[5,6](@ref)
