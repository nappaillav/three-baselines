# no-op stand-in for `setproctitle` on venvs that lack it (e.g. MARLenv, where it only exists as ray's vendored copy
# and is importable only after `import ray`). train.py calls setproctitle.setproctitle(...) purely cosmetically.
def setproctitle(title): pass
def getproctitle(): return ""
