# TrackedCollisionChecker.py
from lecture_examples.IPPerfMonitor import IPPerfMonitor

class TrackedCollisionChecker:
    """ 
    Wraps the standard collision checker to intercept and log 
    collision calls into the IPPerfMonitor dataframe.
    """
    def __init__(self, actual_checker):
        self._actual_checker = actual_checker
        
    @IPPerfMonitor
    def pointInCollision(self, *args, **kwargs):
        return self._actual_checker.pointInCollision(*args, **kwargs)
        
    @IPPerfMonitor
    def lineInCollision(self, *args, **kwargs):
        return self._actual_checker.lineInCollision(*args, **kwargs)
        
    def __getattr__(self, name):
        # Automatically forward any other method calls to the original checker
        return getattr(self._actual_checker, name)