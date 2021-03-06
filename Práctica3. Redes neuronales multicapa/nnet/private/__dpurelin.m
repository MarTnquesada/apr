## Copyright (C) 2005 Michel D. Schmid  <michael.schmid@plexso.com>
##
## This program is free software; you can redistribute it and/or modify it under
## the terms of the GNU General Public License as published by the Free Software
## Foundation; either version 3 of the License, or (at your option) any later
## version.
##
## This program is distributed in the hope that it will be useful, but WITHOUT
## ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
## FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
## details.
##
## You should have received a copy of the GNU General Public License along with
## this program; if not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn {Function File} {} @var{a} = __dpurelin (@var{n})
## @code{dpurelin}, first derivative of purelin
## @example
##
## purelin is a linear transfer function used by neural networks
## @end example
##
## @end deftypefn

## Author: Michel D. Schmid

function a = __dpurelin(n)

   [nRows, nColumns] = size(n);
   a = ones(nRows,nColumns);

endfunction
