## Copyright (C) 2006 Michel D. Schmid   <michael.schmid@plexso.com>
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
## @deftypefn {Function File} {} __printMLPHeader (@var{fid})
## @code{__printMLPHeader} saves the header of a  neural network structure
## to a *.txt file with identification @code{fid}.
## @end deftypefn

## Author: Michel D. Schmid

function __printNetworkType(fid,net)

  if isfield(net,"networkType")
    if strcmp(net.networkType,"newff")
      fprintf(fid,"          Network type:  '%s'\n","Feed forward multi-layer network");
    else
      fprintf(fid,"          Network type:  '%s'\n","error: undefined network type");
    endif
  endif
     
endfunction
