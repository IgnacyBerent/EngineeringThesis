import sys

from loguru import logger

logger.remove()

level_colors = {
    'DEBUG': ('<cyan>', '<fg #b5ffff>'),
    'INFO': ('<green>', '<fg #c9ffd1>'),
    'WARNING': ('<yellow>', '<fg #ffd080>'),
    'ERROR': ('<red>', '<fg #ffa3a3>'),
}


def custom_format(record):
    level = record['level'].name
    strong, light = level_colors.get(level, ('<white>', '<white>'))
    return f'{strong}<bold>{level:<8}</bold></> | {light}{record["message"]}</>\n'


logger.add(sys.stderr, format=custom_format, colorize=True)
