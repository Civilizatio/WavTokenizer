import os
import sys
from pytorch_lightning.callbacks import Callback
import re

# class Tee(object):
#     """类似 Unix tee，stdout/stderr 同时写文件和控制台"""
#     def __init__(self, file, stream, flush_immediately: bool = True):
#         self.file = file
#         self.stream = stream
#         self.flush_immediately = flush_immediately
        
#         self._buffer = ""
        
#         # tqdm or lightning
#         self.__progress_re = re.compile(r"^(Epoch\s*\d+[:\s].*\|)|(\[.*it/s\])")
        

#     def _is_progress_line(self, line: str) -> bool:
#         if self.__progress_re.search(line):
#             return True
        
#         if ("|" in line and "%" in line) or ("it/s" in line and "[" in line):
#             return True
#         return False
    
#     def write(self, data):
#         self.stream.write(data)
#         if self.flush_immediately:
#             self.stream.flush()

#         self.file.write(data)
#         if self.flush_immediately:
#             self.file.flush()
            
#         self._buffer += data
#         while "\n" in self._buffer:
#             line, self._buffer = self._buffer.split("\n", 1)
#             line = line + "\n"
#             # 忽略只有回车/空白的行
#             if line.strip() == "":
#                 continue
#             if self._is_progress_line(line):
#                 # 不写入文件，但保留在控制台
#                 continue
#             self.file.write(line)
#             if self.flush_immediately:
#                 self.file.flush()

#     def flush(self):
#         if self._buffer:
#             if not self._is_progress_line(self._buffer):
#                 self.file.write(self._buffer)
#         self.stream.flush()
#         self.file.flush()
        

class Tee(object):
    """类似 Unix tee，stdout/stderr 同时写文件和控制台"""
    def __init__(self, file, stream, flush_immediately: bool = True):
        self.file = file
        self.stream = stream
        self.flush_immediately = flush_immediately

        self._buffer = ""

        # tqdm/Lightning 进度条或基于回车更新的行
        self.__progress_re = re.compile(
            r"(^Epoch\s*\d+[:\s].*\|)|(\d+/\d+)|(\[.*it/s.*\])|(^Training:)|(^Validation:)"
        )

    def _is_progress_line(self, line: str) -> bool:
        # 识别包含进度条特征或只有 \r 更新的行
        if self.__progress_re.search(line):
            return True
        if ("\r" in line and not "\n" in line) or ("it/s" in line and ("[" in line or "|" in line)):
            return True
        return False

    def write(self, data):
        # 始终把原始输出写到控制台
        try:
            self.stream.write(data)
            if self.flush_immediately:
                self.stream.flush()
        except Exception:
            pass

        # 缓冲所有到达的数据，按行/回车分段处理，只有非进度行写入文件
        self._buffer += data

        # 先处理由 '\r' 分隔的片段（tqdm 常用 '\r' 逐步更新）
        if "\r" in self._buffer:
            parts = self._buffer.split("\r")
            # 所有中间片段视为进度更新，丢弃（但可能包含换行，需要进一步处理）
            for p in parts[:-1]:
                if "\n" in p:
                    # 如果中间片段包含完整行，逐行判断写入
                    for line in p.splitlines(keepends=True):
                        if line.strip() == "":
                            continue
                        if not self._is_progress_line(line):
                            try:
                                self.file.write(line)
                            except Exception:
                                pass
            # 最后那部分作为新的缓冲
            self._buffer = parts[-1]

        # 然后按换行处理剩余完整行
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line + "\n"
            if line.strip() == "":
                continue
            if self._is_progress_line(line):
                # 跳过进度条/频繁更新行
                continue
            try:
                self.file.write(line)
                if self.flush_immediately:
                    self.file.flush()
            except Exception:
                pass

    def flush(self):
        # 把残留缓冲写入（如果不是进度条）
        if self._buffer:
            if not self._is_progress_line(self._buffer):
                try:
                    self.file.write(self._buffer)
                except Exception:
                    pass
            self._buffer = ""
        try:
            self.stream.flush()
            self.file.flush()
        except Exception:
            pass


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

        os.makedirs(log_dir, exist_ok=True)

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
