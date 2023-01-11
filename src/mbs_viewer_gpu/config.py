MIN_IMAGE_HEIGHT = 128
MAX_IMAGE_HEIGHT = 512
RESOLUTION_UPSCALE_FACTOR = 2

STARTING_WINDOW_HEIGHT = 1024

PRECISION_LOG = 3
PRECISION = 2 ** PRECISION_LOG
DOUBLE_PRECISION = PRECISION * 2
FACTOR_PER_CELL = 2 ** 24
DIVMOD_BUFFER = FACTOR_PER_CELL // 2

MIN_NUM_ITERATIONS = 2_000
MAX_NUM_ITERATIONS = 20_000

GRID_WIDTH = 6
GRID_HEIGHT = 6

NUM_ITERATIONS_PER_COLOR_CYCLE = 1000
