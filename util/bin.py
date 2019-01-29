import os.path
import logging
import subprocess


class ExecutionResult(object):
    """
    Holds information about an execution result
    """

    def __init__(self, exit_code, stdout, stderr):
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


class Executable(object):
    """
    Convenience wrapper for subprocess
    """

    def __init__(self, exe):
        """
        Constructor
        :param exe: Path to the executable
        """
        if not os.path.exists(exe):
            raise FileNotFoundError(f'SExtractor binary not found in {exe}')
        self.__exe = exe

    def run(self, *args, **kwargs):
        """
        Execute the program as a separate process, and wait for it.
        :param args: Arguments to be passed as-is to the program.
        :return:
        """
        full_args = [self.__exe] + list(args)
        logging.debug(' '.join([f'"{a}"' for a in full_args]))

        proc = subprocess.Popen(full_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
        stdout, stderr = proc.communicate()
        return ExecutionResult(proc.wait(), stdout.decode('utf-8'), stderr.decode('utf-8'))
