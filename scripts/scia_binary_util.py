#!/usr/bin/env python

import sys
import optparse as op

from sciapy.level1c import scia_limb_scan

convert_options = [
	op.make_option("-a", "--mpl-to-text", action="store_true", dest="mpl_to_text"),
	op.make_option("-A", "--netcdf-to-text", action="store_true", dest="netcdf_to_text"),
	op.make_option("-n", "--text-to-netcdf", action="store_true", dest="text_to_netcdf"),
	op.make_option("-N", "--mpl-to-netcdf", action="store_true", dest="mpl_to_netcdf"),
	op.make_option("-m", "--text-to-mpl", action="store_true", dest="text_to_mpl"),
	op.make_option("-M", "--netcdf-to-mpl", action="store_true", dest="netcdf_to_mpl"),
]
input_options = [
	op.make_option("-f", "--from-type", dest="from_type", choices=["mpl", "netcdf", "text"], default="mpl"),
	op.make_option("-t", "--to-type", dest="to_type", choices=["mpl", "netcdf", "text"], default="text"),
	op.make_option("-i", "--input", dest="input", default=sys.stdin, metavar="FILE"),
	op.make_option("-o", "--output", dest="output", default=sys.stdout, metavar="FILE"),
]
manip_options = [
	op.make_option("-u", "--multiply-by", type=float, dest="mult_factor", default=1.0, metavar="FACTOR"),
	op.make_option("-d", "--add", type=float, dest="add", default=0.0, metavar="NUMBER"),
]

def read_input(sls, rtype, filename):
	if rtype == "mpl":
		sls.read_from_mpl_binary(filename)
	elif rtype == "text":
		sls.read_from_textfile(filename)
	elif rtype == "netcdf":
		sls.read_from_netcdf(filename)

def write_output(sls, wtype, filename):
	if wtype == "mpl":
		sls.write_to_mpl_binary(filename)
	elif wtype == "text":
		sls.write_to_textfile(filename)
	elif wtype == "netcdf":
		sls.write_to_netcdf(filename)


parser = op.OptionParser(option_list=input_options)
convert_group = op.OptionGroup(parser, "Conversion options",
	"Instead of specifying --from-type and --to-type, these options allow"
	"direct conversions between the desired formats.")
for opt in convert_options:
	convert_group.add_option(opt)
parser.add_option_group(convert_group)

manip_group = op.OptionGroup(parser, "Manipulation options",
	"Allows manipulation of the radiance data.")
for opt in manip_options:
	manip_group.add_option(opt)
parser.add_option_group(manip_group)

(options, args) = parser.parse_args()

if options.mpl_to_text:
	options.from_type = "mpl"
	options.to_type = "text"
if options.netcdf_to_text:
	options.from_type = "netcdf"
	options.to_type = "text"
if options.text_to_netcdf:
	options.from_type = "text"
	options.to_type = "netcdf"
if options.mpl_to_netcdf:
	options.from_type = "mpl"
	options.to_type = "netcdf"
if options.text_to_mpl:
	options.from_type = "text"
	options.to_type = "mpl"
if options.netcdf_to_mpl:
	options.from_type = "netcdf"
	options.to_type = "mpl"

slscan = scia_limb_scan()
read_input(slscan, options.from_type, options.input)
#slscan = sn.scia_nadir_scan()
#read_input(slscan, options.from_type, options.input)

if options.mult_factor != 1.0 or options.add != 0.:
	tmp_list = []
	for rad in slscan.rad_list:
		tmp_list.append(rad * options.mult_factor + options.add)
	slscan.rad_list = tmp_list

#slscan.average_spectra()

write_output(slscan, options.to_type, options.output)
