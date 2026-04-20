"""Step 2: SQLite -> qlib bin 格式."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_adapter import AkshareToQlibConverter
from utils.logger import logger


def main():
    conv = AkshareToQlibConverter()
    conv.convert_all()
    conv.build_csi300_instruments()
    logger.info("转换完成, 可在 qlib 中使用: provider_uri='./qlib_data'")


if __name__ == "__main__":
    main()
