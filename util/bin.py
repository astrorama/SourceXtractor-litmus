import logging
import subprocess
import tempfile

try:
    import memory_profiler as mp
except ImportError:
    mp = None
import os.path


class ExecutionResult(object):
    """
    Holds information about an execution result
    """

    def __init__(self, exit_code, stdout, stderr, interval, memory_usage):
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.duration = interval * len(memory_usage)
        self.interval = interval
        self.memory_usage = memory_usage


class Executable(object):
    """
    Convenience wrapper for subprocess
    """

    INTERVAL = .1

    def __init__(self, exe):
        """
        Constructor
        :param exe: Path to the executable
        """
        if not os.path.exists(exe):
            raise FileNotFoundError(f'sourcextractor++ binary not found in {exe}')
        self.__exe = exe

    def run(self, *args, **kwargs):
        """
        Execute the program as a separate process, and wait for it.
        :param args: Arguments to be passed as-is to the program.
        :return:
        """
        full_args = [self.__exe] + list(args)
        logging.debug(' '.join([f'"{a}"' for a in full_args]))

        stdout = tempfile.TemporaryFile('w+b')
        stderr = tempfile.TemporaryFile('w+b')

        proc = subprocess.Popen(full_args, stdout=stdout, stderr=stderr, **kwargs)
        if mp:
            mem = mp.memory_usage(proc=proc, interval=self.INTERVAL, include_children=True)
        else:
            mem = []
            proc.wait()
        stdout.seek(0)
        stderr.seek(0)
        return ExecutionResult(
            proc.wait(), stdout.read().decode('utf-8'), stderr.read().decode('utf-8'),
            self.INTERVAL, mem
        )
