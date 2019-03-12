# Ref: https://github.com/vacancy/Synchronized-BatchNorm-PyTorch


import queue
from collections import namedtuple, OrderedDict
import threading


__all__ = ['FutureResult', 'SlavePipe', 'SyncMaster']


class FutureResult(object):
    """A thread-safe future implementation. Used only as one-to-one pipe."""

    def __init__(self):
        self.result = None
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)

    def put(self, result):
        with self.lock:
            assert self.result is None, 'Previous result has\'t been fetched.'
            self.result = result
            self.cond.notify()

    def get(self):
        with self.lock:
            if self.result is None:
                self.cond.wait()
            
            res = self.result
            self.result = None
            return res


_MasterRegistry = namedtuple('MasterRegistry', ['result'])
_SlavePipeBase = namedtuple('_SlavePipeBase', ['identifier', 'quene', 'result'])


class SlavePipe(_SlavePipeBase):
    """Pipe for master-slave communication."""

    def run_slave(self, msg):
        self.quene.put((self.identifier, msg))
        ret = self.result.get()
        self.quene.put(True)
        return ret


class SyncMaster(object):

    def __init__(self, master_callback):
        """
        Args:
            master_callback (): a callback to be invoked after having collencted messages from slave devices.
        """
        self.master_callback = master_callback
        self.quene = queue.Queue()
        self.registry = OrderedDict()
        self.activated = False


    def __getstate__(self):
        return {'master_callback': self.master_callback}

    
    def __setstate__(self, state):
        self.__init__(state['master_callback'])
    

    def register_slave(self, identifier):
        """Register an slave device.
        Args:
            identifier (): an identifier, usually is the device id.
        Returns (): a `SlavePipe` object which can be used to communicati with the master device.
        """
        if self.activated:
            assert self.quene.empty(), 'Quene is not clean before next initialization.'
            self.activated = False
            self.registry.clear()
        future = FutureResult()
        self.registry[identifier] = _MasterRegistry(future)
        return SlavePipe(identifier, self.quene, future)

    
    def run_master(self, master_msg):
        """
        Args:
            master_msg (): 
        """
        self.activated = True

        intermediates = [(0, master_msg)]
        for _ in range(self.num_slaves):
            intermediates.append(self.quene.get())
        
        results = self.master_callback(intermediates)
        assert results[0][0] == 0, 'The first result should belong to the master.'

        for i, res in results:
            if i == 0:
                continue
            self.registry[i].result.put(res)

        for _ in range(self.num_slaves):
            assert self.quene.get() is True
        
        return results[0][1]
        

    @property
    def num_slaves(self):
        return len(self.registry)