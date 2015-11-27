Python 2.7.10 (v2.7.10:15c95b7d81dc, May 23 2015, 09:33:12) 
[GCC 4.2.1 (Apple Inc. build 5666) (dot 3)] on darwin
Type "copyright", "credits" or "license()" for more information.
>>> WARNING: The version of Tcl/Tk (8.5.9) in use may be unstable.
Visit http://www.python.org/download/mac/tcltk/ for current information.
================================ RESTART ================================
>>> 
>>> x = [1,2,3]
>>> y = [0,0,0]
>>> z = [0,0,0]
>>> dx = [0.5, 0.5, 0.5]
>>> dy = [0.5, 0.5, 0.5]
>>> dz = [5, 4, 7]
>>> bar3d(x, y, z, dx, dy, dz, color='b', zsort='average', *args, **kwargs)

Traceback (most recent call last):
  File "<pyshell#6>", line 1, in <module>
    bar3d(x, y, z, dx, dy, dz, color='b', zsort='average', *args, **kwargs)
NameError: name 'bar3d' is not defined
>>> from matplotlib
SyntaxError: invalid syntax
>>> import matplotlib
>>> ================================ RESTART ================================
>>> 
>>> ================================ RESTART ================================
>>> 
red 0
blue 6.87405285471
red 0
blue 0.283038704963
m 0
c 5.95932543072
red 12.6291128044
blue 12.7291128044
m 0
c 0.1
red 0.2
blue 6.56832535152
y 6.66832535152
>>> 
