import argparse
import logging
import time
import pandas as pd
import akshare as ak
from logging.handlers import RotatingFileHandler
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import chardet
import numpy as np
import sys
import importlib.util
from datetime import date
from pathlib import Path

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from etf_daily.paths import FUND_CRAWLER_LOG_DIR, FUND_LIST_CSV, QLIB_CHANGE_CSV_DIR, QLIB_PROVIDER_URI, QLIB_SCRIPTS_DIR
from etf_daily.etl.sina_helpers import (
    calendars_day_exists,
    choose_dump_mode,
    csv_max_date,
    merge_ohlcv,
    polite_delay_seconds,
    should_skip_fund,
)


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


def resolve_dump_bin_path(qlib_scripts_path: str) -> str:
    dump_bin_path = os.path.join(qlib_scripts_path, 'dump_bin.py')
    if not os.path.exists(dump_bin_path):
        raise FileNotFoundError(
            "未找到 QLib 转换脚本："
            f"{dump_bin_path}。请检查 configs/production_regime_switch_ewma_shrink.json "
            "中的 paths.qlib_scripts_dir 是否指向本机 qlib 源码目录。"
        )
    return dump_bin_path

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


def fetch_fund_data_with_retry(
        symbol: str,
        start_date,
        end_date,
        max_retries: int = 3,
        base_timeout: int = 10,
        backoff_factor: float = 1.5
):
    """
    带重试机制的基金数据抓取
    :param symbol: 基金代码
    :param max_retries: 最大重试次数(默认3次)
    :param base_timeout: 基础超时时间(秒)
    :param backoff_factor: 退避因子(指数退避)
    """
    current_timeout = base_timeout
    for attempt in range(max_retries):
        try:
            # 动态设置超时时间（网页1的优化建议）
            return ak.fund_etf_hist_sina(symbol=symbol)
        except ak.exceptions.TimeoutError as e:
            logger.warning(f"基金 {symbol} 第{attempt + 1}次请求超时（超时时间：{current_timeout}s）")
            if attempt == max_retries - 1:
                raise  # 达到最大重试次数后抛出异常

            # 指数退避策略（网页2的断路器优化思路）
            sleep_time = backoff_factor ** attempt
            time.sleep(sleep_time)
            current_timeout = int(current_timeout * 1.2)  # 每次增加20%超时时长

        except Exception as e:
            logger.error(f"基金 {symbol} 发生非超时异常: {str(e)}", exc_info=True)
            raise

    return None  # 所有重试失败返回空值


def fetch_and_save_data_parallel(
    dir_name,
    start_date,
    end_date,
    *,
    as_of: str,
    full: bool = False,
    workers: int = 8,
    polite_delay: bool = False,
    progress_every: int = 50,
):
    """并行化基金数据抓取。默认跳过本地 CSV 已覆盖 as_of 的基金。"""
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
        skipped_count = 0
        failed_codes = []
        codes = codefundsecname['基金代码'].tolist()
        done_count = 0

        # Ensure target directory exists (dir_name may contain $home placeholder)
        dir_name = normalize_path(dir_name)
        os.makedirs(dir_name, exist_ok=True)

        max_workers = max(1, int(workers))
        logger.info(
            f"抓取模式: {'full' if full else 'incremental'} as_of={as_of} "
            f"workers={max_workers} polite_delay={polite_delay}"
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_single_fund,
                    code=code,
                    idx=idx + 1,
                    total=total_count,
                    dir_name=dir_name,
                    start_date=start_date,
                    end_date=end_date,
                    as_of=as_of,
                    full=full,
                    polite_delay=polite_delay,
                ): code for idx, code in enumerate(codes)
            }

            for future in as_completed(futures):
                code = futures[future]
                done_count += 1
                try:
                    result = future.result()
                    if result == "skipped":
                        skipped_count += 1
                        success_count += 1
                    elif result:
                        success_count += 1
                        logger.debug(f"基金 {code} 处理完成")
                except Exception as e:
                    logger.error(f"基金 {code} 处理失败: {str(e)}", exc_info=True)
                    failed_codes.append(code)

                if progress_every > 0 and (
                    done_count % progress_every == 0 or done_count == total_count
                ):
                    logger.info(
                        f"进度 {done_count}/{total_count} "
                        f"ok={success_count} skipped={skipped_count} failed={len(failed_codes)}"
                    )

        logger.info(
            f"任务完成：成功 {success_count}/{total_count} "
            f"(skipped={skipped_count}, failed={len(failed_codes)})"
        )
        if failed_codes:
            logger.warning(f"失败基金代码列表：{failed_codes}")

    except Exception as e:
        logger.critical("主程序异常终止", exc_info=True)
        raise


def process_single_fund(
    code,
    idx,
    total,
    dir_name,
    start_date,
    end_date,
    *,
    as_of: str,
    full: bool = False,
    polite_delay: bool = False,
):
    """单支基金处理流程。返回 True / False / \"skipped\"。"""
    file_path = os.path.join(dir_name, f"{code}.csv")
    try:
        logger.debug(f"开始处理基金 {code} ({idx}/{total})")

        if not full:
            local_max = csv_max_date(file_path)
            if should_skip_fund(local_max, as_of):
                logger.debug(f"基金 {code} 已覆盖 as_of={as_of} (max={local_max})，跳过")
                return "skipped"

        fund_df = fetch_fund_data_with_retry(
            code, start_date, end_date,
            max_retries=5,
            base_timeout=15
        )

        if fund_df is None or fund_df.empty:
            logger.warning(f"基金 {code} 无有效数据")
            return False

        fund_df = fund_df.rename(columns={
            'date': 'date', 'open': 'open', 'close': 'close',
            'high': 'high', 'low': 'low', 'volume': 'volume'
        })
        fund_df["date"] = pd.to_datetime(fund_df['date']).dt.strftime('%Y-%m-%d')
        fund_df["code"] = code

        if not full and os.path.isfile(file_path):
            try:
                old_df = pd.read_csv(file_path)
                fund_df = merge_ohlcv(old_df, fund_df)
            except Exception as e:
                logger.warning(f"基金 {code} 合并旧 CSV 失败，改为全量覆盖: {e}")

        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        fund_df.to_csv(file_path, index=False)
        logger.debug(f"基金 {code} 数据已保存至 {file_path}")
        return True

    except Exception:
        raise
    finally:
        delay = polite_delay_seconds(polite_delay)
        if delay > 0:
            time.sleep(delay)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sina ETF ETL → CSV + qlib dump_bin")
    parser.add_argument(
        "--as-of",
        default=date.today().isoformat(),
        help="Skip funds whose local CSV max date >= this (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Re-fetch and overwrite all funds; use dump_all.",
    )
    parser.add_argument("--workers", type=int, default=8, help="ThreadPool workers (default 8).")
    parser.add_argument(
        "--polite-delay",
        action="store_true",
        help="Opt-in 1–3s sleep after each fund (default off).",
    )
    parser.add_argument(
        "--skip-dump",
        action="store_true",
        help="Fetch/write CSVs only; skip qlib dump_bin.",
    )
    parser.add_argument("--start-date", default="20050101", help="Legacy fetch start (YYYYMMDD).")
    parser.add_argument("--end-date", default="20301202", help="Legacy fetch end (YYYYMMDD).")
    return parser.parse_args(argv)


def run_dump_bin(*, csv_path: str, qlib_dir: str, dump_mode: str) -> None:
    if QLIB_SCRIPTS_DIR is None:
        raise SystemExit("qlib_scripts_dir is unset. Set paths.qlib_scripts_dir in the local json.")
    qlib_scripts_path = normalize_path(str(QLIB_SCRIPTS_DIR))
    dump_bin_path = resolve_dump_bin_path(qlib_scripts_path)

    for path in [csv_path, qlib_dir]:
        if not os.path.exists(path):
            logger.warning(f"路径不存在：{path}")

    python_exe = sys.executable
    if importlib.util.find_spec('fire') is None:
        logger.error(
            "缺少依赖模块 'fire'。请在运行脚本的 Python 环境中安装它，例如：\n"
            f"{python_exe} -m pip install fire"
        )
        raise RuntimeError("缺少依赖模块 'fire'，已在日志中给出安装命令。")

    command = [
        python_exe,
        dump_bin_path,
        dump_mode,
        '--data_path', csv_path,
        '--qlib_dir', qlib_dir,
        '--symbol_field_name', 'code',
        '--date_field_name', 'date',
        '--include_fields', 'open,high,low,close,volume',
    ]
    logger.info(f"运行 dump_bin: {dump_mode}")
    try:
        subprocess.run(command, check=True, capture_output=True)
        logger.info("命令执行成功！")
    except subprocess.CalledProcessError as e:
        detected_encoding = chardet.detect(e.stderr)['encoding'] or 'utf-8'
        decoded_stderr = e.stderr.decode(detected_encoding, errors='replace')
        logger.error(f"命令执行失败，错误码：{e.returncode}")
        logger.error("错误信息：")
        logger.error(decoded_stderr)
        raise
    except KeyboardInterrupt:
        logger.warning("程序被用户中断（Ctrl+C）")
        logger.info("清理完成，程序退出")
        raise
    finally:
        logger.info("执行最终清理")


if __name__ == '__main__':
    try:
        args = parse_args()
        logger.info("程序启动")
        csv_path = normalize_path(str(QLIB_CHANGE_CSV_DIR))
        qlib_dir = normalize_path(str(QLIB_PROVIDER_URI))

        fetch_and_save_data_parallel(
            csv_path,
            args.start_date,
            args.end_date,
            as_of=args.as_of,
            full=args.full,
            workers=args.workers,
            polite_delay=args.polite_delay,
        )
        logger.info("程序正常退出")

        if args.skip_dump:
            logger.info("--skip-dump: 跳过 qlib dump_bin")
        else:
            dump_mode = choose_dump_mode(args.full, calendars_day_exists(qlib_dir))
            run_dump_bin(csv_path=csv_path, qlib_dir=qlib_dir, dump_mode=dump_mode)

    except Exception as e:
        logger.exception("未捕获的全局异常")  # 自动记录堆栈跟踪[5,6](@ref)
        raise SystemExit(1) from e
