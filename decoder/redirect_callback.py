import os
import sys
from pytorch_lightning.callbacks import Callback

class Tee(object):
    """类似 Unix tee，stdout/stderr 同时写文件和控制台"""
    def __init__(self, file, stream, flush_immediately: bool = True):
        self.file = file
        self.stream = stream
        self.flush_immediately = flush_immediately

    def write(self, data):
        self.stream.write(data)
        if self.flush_immediately:
            self.stream.flush()

        self.file.write(data)
        if self.flush_immediately:
            self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

class RedirectStdCallback(Callback):
    def __init__(self, filename: str = "train.log", flush_immediately: bool = True):
        super().__init__()
        self.filename = filename
        self.flush_immediately = flush_immediately
        self._orig_stdout = None
        self._orig_stderr = None
        self._log_file = None

    def on_fit_start(self, trainer, pl_module):
        log_dir = trainer.logger.log_dir if trainer.logger is not None else "."
        log_path = os.path.join(log_dir, self.filename)

        # 打开 log 文件
        self._log_file = open(log_path, "w")

        # 备份原始 stdout/stderr
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr

        # 重定向到 tee
        sys.stdout = Tee(self._log_file, self._orig_stdout, self.flush_immediately)
        sys.stderr = Tee(self._log_file, self._orig_stderr, self.flush_immediately)

        print(f"[RedirectStdCallback] 日志记录到 {log_path}")

    def on_fit_end(self, trainer, pl_module):
        # 恢复 stdout/stderr
        if self._log_file:
            sys.stdout = self._orig_stdout
            sys.stderr = self._orig_stderr
            self._log_file.close()
            self._log_file = None
