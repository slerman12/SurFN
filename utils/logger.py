import tonic


current_logger = None


def initialize_clearml_logger(clearml_logger):
    class TrainsLogger(tonic.logger.Logger):
        def store(self, key, value, stats=False):
            super().store(key, value, stats)
            # todo add trains logging
            return clearml_logger

    def _initialize(*args, **kwargs):
        global current_logger
        current_logger = TrainsLogger(*args, **kwargs)
        return current_logger

    def _get_current_logger():
        global current_logger
        if current_logger is None:
            current_logger = TrainsLogger()
        return current_logger

    return _initialize, _get_current_logger


def initialize(clearml_logger):
    tonic.logger.initialize, tonic.logger.get_current_logger = initialize_clearml_logger(clearml_logger)
