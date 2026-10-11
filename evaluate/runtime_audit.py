"""Exclusive, reversible ownership of a process-global runtime hook."""
from threading import RLock

_ACTIVE = set()
_LOCK = RLock()


class RuntimePatch:
    def __init__(self, target, attribute, replacement):
        self.target,self.attribute = target,attribute
        self.key = (id(target),attribute)
        self.replacement = replacement
        self.active = False

    def __enter__(self):
        with _LOCK:
            if self.active or self.key in _ACTIVE:
                raise RuntimeError('Runtime hook already has an active owner')
            self.original = getattr(self.target,self.attribute)
            setattr(self.target,self.attribute,self.replacement)
            _ACTIVE.add(self.key)
            self.active = True
        return self

    def close(self):
        with _LOCK:
            if self.active:
                setattr(self.target,self.attribute,self.original)
                _ACTIVE.remove(self.key)
                self.active = False

    def __exit__(self, *exception):
        self.close()
