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
            current_timeout = int(current_timeout * 1.2)  # 每次增加20%超时阈值

        except Exception as e:
            logger.error(f"基金 {symbol} 发生非超时异常: {str(e)}", exc_info=True)
            raise

    return None  # 所有重试失败返回空值


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
        dump_bin_path = resolve_dump_bin_path(qlib_scripts_path)

        # 检查路径是否存在（参考[1,5](@ref)）
        for path in [csv_path, qlib_dir]:
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
            dump_bin_path,
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
