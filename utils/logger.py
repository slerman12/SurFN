import tonic

current_logger = None


def init_clearml_logger(clearml_logger):
    class ClearmlLogger(tonic.logger.Logger):
        def store(self, key, value, stats=False):
            super().store(key, value, stats)
            clearml_logger.report_scalar(key, key, value=value)

    def _initialize(*args, **kwargs):
        global current_logger
        current_logger = ClearmlLogger(*args, **kwargs)
        return current_logger

    def _get_current_logger():
        global current_logger
        if current_logger is None:
            current_logger = ClearmlLogger()
        return current_logger

    def _dump(*args, **kwargs):
        # make_scatter = lambda x: list(zip(*x))
        # clearml_logger.report_scatter2d("Avg Head Size/Avg Num Explored", "series_markers", iteration=episode,
        #                                 scatter=make_scatter([avg_head_sizes, avg_num_explored]),
        #                                 xaxis="Avg Head Size", yaxis="Avg Num Explored", mode='markers')
        logger = _get_current_logger()
        return logger.dump(*args, **kwargs)

    return _initialize, _get_current_logger, _dump


def initialize(clearml_logger):
    tonic.logger.initialize, tonic.logger.get_current_logger, tonic.logger.dump = init_clearml_logger(clearml_logger)
