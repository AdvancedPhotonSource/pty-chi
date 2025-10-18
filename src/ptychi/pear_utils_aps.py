import sys
from tkinter import filedialog as tkFileDialog
import tkinter as Tkinter
import os
import string

if sys.version_info < (3, 13):
    from xdrlib import *
else:
    from xdrlib3 import *
	
class scanDim:
	def __init__(self):
		self.rank = 0
		self.dim = 0
		self.npts = 0
		self.curr_pt = 0
		self.scan_name = ""
		self.time = ""
		self.np = 0
		self.p = []				# list of scanPositioner instances
		self.nd = 0
		self.d = []				# list of scanDetector instances
		self.nt = 0
		self.t = []				# list of scanTrigger instances

	def __str__(self):
		if self.scan_name != '':
			s = "%dD data from \"%s\": %d/%d pts; %d pos\'s, %d dets, %d trigs" % (
				self.dim, self.scan_name, self.curr_pt, self.npts, self.np, self.nd,
				self.nt)
		else:
			s = "%dD data (not read in)" % (self.dim)

		return s

class scanClass:
	def __init__(self):
		self.rank = 0
		self.npts = 0
		self.curr_pt = 0
		self.plower_scans = 0
		self.name = ""
		self.time = ""
		self.np = 0
		self.nd = 0
		self.nt = 0
		self.p = []
		self.d = []
		self.t = []

class scanPositioner:
	def __init__(self):
		self.number = 0
		self.fieldName = ""
		self.name = ""
		self.desc = ""
		self.step_mode = ""
		self.unit = ""
		self.readback_name = ""
		self.readback_desc = ""
		self.readback_unit = ""
		self.data = []

	def __str__(self):
		s = "positioner %d (%s), desc:%s, unit:%s\n" % (self.number, self.name,
			self.desc, self.unit)
		s = s + "   step mode: %s, readback:\"%s\"\n" % (self.step_mode,
			self.readback_name)
		s = s + "data:%s" % (str(self.data))
		return s

class scanDetector:
	def __init__(self):
		self.number = 0
		self.fieldName = ""
		self.name = ""
		self.desc = ""
		self.unit = ""
		self.data = []

	def __str__(self):
		s = "detector %d (%s), desc:%s, unit:%s, data:%s\n" % (self.number,
			self.name, self.desc, self.unit, str(self.data))
		return s

class scanTrigger:
	def __init__(self):
		self.number = 0
		self.name = ""
		self.command = 0.0

	def __str__(self):
		s = "trigger %d (%s), command=%f\n" % (self.number,
			self.name, self.command)
		return s


def detName(i,new=0):
	"""
	detName(i,new=0) - this function returns the detector name Di used in sscan record
	where 
	  i - specify the zero based detector sequence number 
	  new - 1 specify the version 5 Di names desired, default 0
	"""
	if new:
		return "D%02d"%(i+1)
	if i < 15:
		return ("D%s"%(hex(i+1)[2])).upper()
	elif i < 85:
		return "D%02d"%(i-14)
	else:
		return "?"

def posName(i):
	"""
	posName(i) - this function returns the positioner name Pi used in sscan record
	where 
	  i - specify the zero based positioner sequence number 
	"""
	if i < 4:
		return "P%d" % (i+1)
	else:
		return "?"

def readScan(file, v, new=0):
	"""
	readScan(file, v, new=0) - internal scan read routine, it unpack a subset of scan data from
	the current position of the file pointer 
	it returns the scan data set extracted
	where
	  file - file pointer of an opened MDA file
	  v - input verbose specified
	  new  - default 0, if 1 specified then version 5 Di name used
	"""
	scan = scanClass()
	buf = file.read(10000) # enough to read scan header
	u = Unpacker(buf)
	scan.rank = u.unpack_int()
	if v: print("scan.rank = ", repr(scan.rank))
	scan.npts = u.unpack_int()
	if v: print("scan.npts = ", repr(scan.npts))
	scan.curr_pt = u.unpack_int()
	if v: print("scan.curr_pt = ", repr(scan.curr_pt))
	if (scan.rank > 1):
		# if curr_pt < npts, plower_scans will have garbage for pointers to
		# scans that were planned for but not written
		scan.plower_scans = u.unpack_farray(scan.npts, u.unpack_int)
		if v: print("scan.plower_scans = ", repr(scan.plower_scans))
	namelength = u.unpack_int()
	scan.name = u.unpack_string()
	if v: print("scan.name = ", repr(scan.name))
	timelength = u.unpack_int()
	scan.time = u.unpack_string()
	if v: print("scan.time = ", repr(scan.time))
	scan.np = u.unpack_int()
	if v: print("scan.np = ", repr(scan.np))
	scan.nd = u.unpack_int()
	if v: print("scan.nd = ", repr(scan.nd))
	scan.nt = u.unpack_int()
	if v: print("scan.nt = ", repr(scan.nt))
	for j in range(scan.np):
		scan.p.append(scanPositioner())
		scan.p[j].number = u.unpack_int()
		scan.p[j].fieldName = posName(scan.p[j].number)
		if v: print("positioner ", j)
		length = u.unpack_int() # length of name string
		if length: scan.p[j].name = u.unpack_string()
		if v: print("scan.p[%d].name = %s" % (j, repr(scan.p[j].name)))
		length = u.unpack_int() # length of desc string
		if length: scan.p[j].desc = u.unpack_string()
		if v: print("scan.p[%d].desc = %s" % (j, repr(scan.p[j].desc)))
		length = u.unpack_int() # length of step_mode string
		if length: scan.p[j].step_mode = u.unpack_string()
		if v: print("scan.p[%d].step_mode = %s" % (j, repr(scan.p[j].step_mode)))
		length = u.unpack_int() # length of unit string
		if length: scan.p[j].unit = u.unpack_string()
		if v: print("scan.p[%d].unit = %s" % (j, repr(scan.p[j].unit)))
		length = u.unpack_int() # length of readback_name string
		if length: scan.p[j].readback_name = u.unpack_string()
		if v: print("scan.p[%d].readback_name = %s" % (j, repr(scan.p[j].readback_name)))
		length = u.unpack_int() # length of readback_desc string
		if length: scan.p[j].readback_desc = u.unpack_string()
		if v: print("scan.p[%d].readback_desc = %s" % (j, repr(scan.p[j].readback_desc)))
		length = u.unpack_int() # length of readback_unit string
		if length: scan.p[j].readback_unit = u.unpack_string()
		if v: print("scan.p[%d].readback_unit = %s" % (j, repr(scan.p[j].readback_unit)))

	for j in range(scan.nd):
		scan.d.append(scanDetector())
		scan.d[j].number = u.unpack_int()
		scan.d[j].fieldName = detName(scan.d[j].number,new=new)
		if v: print("detector ", j)
		length = u.unpack_int() # length of name string
		if length: scan.d[j].name = u.unpack_string()
		if v: print("scan.d[%d].name = %s" % (j, repr(scan.d[j].name)))
		length = u.unpack_int() # length of desc string
		if length: scan.d[j].desc = u.unpack_string()
		if v: print("scan.d[%d].desc = %s" % (j, repr(scan.d[j].desc)))
		length = u.unpack_int() # length of unit string
		if length: scan.d[j].unit = u.unpack_string()
		if v: print("scan.d[%d].unit = %s" % (j, repr(scan.d[j].unit)))

	for j in range(scan.nt):
		scan.t.append(scanTrigger())
		scan.t[j].number = u.unpack_int()
		if v: print("trigger ", j)
		length = u.unpack_int() # length of name string
		if length: scan.t[j].name = u.unpack_string()
		if v: print("scan.t[%d].name = %s" % (j, repr(scan.t[j].name)))
		scan.t[j].command = u.unpack_float()
		if v: print("scan.t[%d].command = %s" % (j, repr(scan.t[j].command)))

	### read data
	# positioners
	file.seek(file.tell() - (len(buf) - u.get_position()))
	buf = file.read(scan.np * scan.npts * 8)
	u = Unpacker(buf)
	for j in range(scan.np):
		if v: print("read %d pts for pos. %d at buf loc %x" % (scan.npts,
			j, u.get_position()))
		scan.p[j].data = u.unpack_farray(scan.npts, u.unpack_double)    
		if v: print("scan.p[%d].data = %s" % (j, repr(scan.p[j].data)))
        
	# detectors
	file.seek(file.tell() - (len(buf) - u.get_position()))
	buf = file.read(scan.nd * scan.npts * 4)
	u = Unpacker(buf)
	for j in range(scan.nd):
		scan.d[j].data = u.unpack_farray(scan.npts, u.unpack_float)
		if v: print("scan.d[%d].data = %s" % (j, repr(scan.d[j].data)))

	return scan

def readMDA(fname=None, maxdim=2, verbose=1, help=0, new=0):
	"""
	readMDA(fname=None, maxdim=2, verbose=1, help=0) - This fuction read an MDA file and 
	construct the MDA data structure accordingly 
	it returns the MDA data sturcture constructed
	where
	  fname - specifies the input mda file name
	  maxdim - specifies the max dimension extract, default 2 
	  verbose - reading info on or off, default 1
	  help - echo help information on or off, default 0 
	  new - 1 specify the version 5 Di names desired, default 0

	e.g.

	from readMDA import *
	d = readMDA('/home/beams/CHA/data/xxx/cha_0001.mda')

	"""
	dim = []

	if (fname == None):
		fname = tkFileDialog.Open().show()
	if (not os.path.isfile(fname)): fname = fname + '.mda'
	if (not os.path.isfile(fname)):
		print(fname, "is not a file")
		return dim

	file = open(fname, 'rb')
	#print "file = ", str(file)
	#file.seek(0,2)
	#filesize = file.tell()
	#file.seek(0)
	buf = file.read(100)		# to read header for scan of up to 5 dimensions
	u = Unpacker(buf)

	# read file header
	version = u.unpack_float()
	scan_number = u.unpack_int()
	rank = u.unpack_int()
	dimensions = u.unpack_farray(rank, u.unpack_int)
	isRegular = u.unpack_int()
	pExtra = u.unpack_int()
	pmain_scan = file.tell() - (len(buf) - u.get_position())

	for i in range(rank):
		dim.append(scanDim())
		dim[i].dim = i+1
		dim[i].rank = rank-i

	file.seek(pmain_scan)
	s0 = readScan(file, max(0,verbose-1),new=new)
	dim[0].npts = s0.npts
	dim[0].curr_pt = s0.curr_pt
	dim[0].scan_name = s0.name
	dim[0].time = s0.time
	dim[0].np = s0.np
	for i in range(s0.np): dim[0].p.append(s0.p[i])
	dim[0].nt = s0.nt
	for j in range(s0.nt): dim[0].t.append(s0.t[j])
	dim[0].nd = s0.nd
	for i in range(s0.nd): dim[0].d.append(s0.d[i])

	if ((rank > 1) and (maxdim > 1)):
		# collect 2D data
		for i in range(s0.curr_pt):
			file.seek(s0.plower_scans[i])
			s = readScan(file, max(0,verbose-1),new=new)
			if i == 0:
				dim[1].npts = s.npts
				dim[1].curr_pt = s.curr_pt
				dim[1].scan_name = s.name
				dim[1].time = s.time
				# copy positioner, trigger, detector instances
				dim[1].np = s.np
				for j in range(s.np):
					dim[1].p.append(s.p[j])
					tmp = s.p[j].data[:]
					dim[1].p[j].data = []
					dim[1].p[j].data.append(tmp)
				dim[1].nt = s.nt
				for j in range(s.nt): dim[1].t.append(s.t[j])
				dim[1].nd = s.nd
				for j in range(s.nd):
					dim[1].d.append(s.d[j])
					tmp = s.d[j].data[:]
					dim[1].d[j].data = []
					dim[1].d[j].data.append(tmp)
			else:
				# append data arrays
				for j in range(s.np): dim[1].p[j].data.append(s.p[j].data)
				for j in range(s.nd): dim[1].d[j].data.append(s.d[j].data)

	if ((rank > 2) and (maxdim > 2)):
		# collect 3D data
		for i in range(s0.curr_pt):
			file.seek(s0.plower_scans[i])
			s1 = readScan(file, max(0,verbose-1),new=new)
			for j in range(s1.curr_pt):
				file.seek(s1.plower_scans[j])
				s = readScan(file, max(0,verbose-1),new=new)
				if ((i == 0) and (j == 0)):
					dim[2].npts = s.npts
					dim[2].curr_pt = s.curr_pt
					dim[2].scan_name = s.name
					dim[2].time = s.time
					# copy positioner, trigger, detector instances
					dim[2].np = s.np
					for k in range(s.np):
						dim[2].p.append(s.p[k])
						tmp = s.p[k].data[:]
						dim[2].p[k].data = [[]]
						dim[2].p[k].data[i].append(tmp)
					dim[2].nt = s.nt
					for k in range(s.nt): dim[2].t.append(s.t[k])
					dim[2].nd = s.nd
					for k in range(s.nd):
						dim[2].d.append(s.d[k])
						tmp = s.d[k].data[:]
						dim[2].d[k].data = [[]]
						dim[2].d[k].data[i].append(tmp)
				elif j == 0:
					for k in range(s.np):
						dim[2].p[k].data.append([])
						dim[2].p[k].data[i].append(s.p[k].data)
					for k in range(s.nd):
						dim[2].d[k].data.append([])
						dim[2].d[k].data[i].append(s.d[k].data)
				else:
					# append data arrays
					for k in range(s.np): dim[2].p[k].data[i].append(s.p[k].data)
					for k in range(s.nd): dim[2].d[k].data[i].append(s.d[k].data)

	# Collect scan-environment variables into a dictionary
	dict = {}
	dict['sampleEntry'] = ("description", "unit string", "value")
	dict['filename'] = fname
	dict['rank'] = rank
	dict['dimensions'] = dimensions
	if pExtra:
		file.seek(pExtra)
		buf = file.read()       # Read all scan-environment data
		u = Unpacker(buf)
		numExtra = u.unpack_int()
		for i in range(numExtra):
			name = ''
			n = u.unpack_int()      # length of name string
			if n: name = u.unpack_string()
			desc = ''
			n = u.unpack_int()      # length of desc string
			if n: desc = u.unpack_string()
			type = u.unpack_int()

			unit = ''
			value = ''
			count = 0
			if type != 0:   # not DBR_STRING
				count = u.unpack_int()  # 
				n = u.unpack_int()      # length of unit string
				if n: unit = u.unpack_string()

			if type == 0: # DBR_STRING
				n = u.unpack_int()      # length of value string
				if n: value = u.unpack_string()
			elif type == 32: # DBR_CTRL_CHAR
				#value = u.unpack_fstring(count)
				v = u.unpack_farray(count, u.unpack_int)
				value = ""
				for i in range(len(v)):
					# treat the byte array as a null-terminated string
					if v[i] == 0: break
					value = value + chr(v[i])

			elif type == 29: # DBR_CTRL_SHORT
				value = u.unpack_farray(count, u.unpack_int)
			elif type == 33: # DBR_CTRL_LONG
				value = u.unpack_farray(count, u.unpack_int)
			elif type == 30: # DBR_CTRL_FLOAT
				value = u.unpack_farray(count, u.unpack_float)
			elif type == 34: # DBR_CTRL_DOUBLE
				value = u.unpack_farray(count, u.unpack_double)

			dict[name] = (desc, unit, value)

	dim.reverse()
	dim.append(dict)
	dim.reverse()
	if verbose:
		print("%s is a %d-D file; %d dimensions read in." % (fname, dim[0]['rank'], len(dim)-1))
		print("dim[0] = dictionary of %d scan-environment PVs" % (len(dim[0])))
		print("   usage: dim[0]['sampleEntry'] ->", dim[0]['sampleEntry'])
		for i in range(1,len(dim)):
			print("dim[%d] = %s" % (i, str(dim[i])))
		print("   usage: dim[1].p[2].data -> 1D array of positioner 2 data")
		print("   usage: dim[2].d[7].data -> 2D array of detector 7 data")

	if help:
		print(" ")
		print("   each dimension (e.g., dim[1]) has the following fields: ")
		print("      time      - date & time at which scan was started: %s" % (dim[1].time))
		print("      scan_name - name of scan record that acquired this dimension: '%s'" % (dim[1].scan_name))
		print("      curr_pt   - number of data points actually acquired: %d" % (dim[1].curr_pt))
		print("      npts      - number of data points requested: %d" % (dim[1].npts))
		print("      nd        - number of detectors for this scan dimension: %d" % (dim[1].nd))
		print("      d[]       - list of detector-data structures")
		print("      np        - number of positioners for this scan dimension: %d" % (dim[1].np))
		print("      p[]       - list of positioner-data structures")
		print("      nt        - number of detector triggers for this scan dimension: %d" % (dim[1].nt))
		print("      t[]       - list of trigger-info structures")

	if help:
		print(" ")
		print("   each detector-data structure (e.g., dim[1].d[0]) has the following fields: ")
		print("      desc      - description of this detector")
		print("      data      - data list")
		print("      unit      - engineering units associated with this detector")
		print("      fieldName - scan-record field (e.g., 'D01')")


	if help:
		print(" ")
		print("   each positioner-data structure (e.g., dim[1].p[0]) has the following fields: ")
		print("      desc          - description of this positioner")
		print("      data          - data list")
		print("      step_mode     - scan mode (e.g., Linear, Table, On-The-Fly)")
		print("      unit          - engineering units associated with this positioner")
		print("      fieldName     - scan-record field (e.g., 'P1')")
		print("      name          - name of EPICS PV (e.g., 'xxx:m1.VAL')")
		print("      readback_desc - description of this positioner")
		print("      readback_unit - engineering units associated with this positioner")
		print("      readback_name - name of EPICS PV (e.g., 'xxx:m1.VAL')")

	return dim


def pickMDA():
	"""
	pickMDA() - This function lets user use file selection dialog to pick desired mda file,
	then passes the selected file to the readMDA function 
	it returns the MDA data structure constructed

	e.g.

	from readMDA import *
	d = pickMDA()
	"""
	root = Tkinter.Tk()
	if len(sys.argv) < 2:
		fname = tkFileDialog.Open().show()
	elif sys.argv[1] == '?' or sys.argv[1] == "help" or sys.argv[1][:2] == "-h":
		print("usage: %s [filename [maxdim [verbose]]]" % sys.argv[0])
		print("   maxdim defaults to 2; verbose defaults to 1")
		return()
	else:
		fname = sys.argv[1]
	if fname == (): return

	maxdim = 2
	verbose = 1
	if len(sys.argv) > 1:
		maxdim = int(sys.argv[2])
	if len(sys.argv) > 2:
		verbose = int(sys.argv[3])
	if len(sys.argv) > 3:
		help = int(sys.argv[4])
	else:  
		help=0

	dim = readMDA(fname, maxdim, verbose, help)
	return dim

if __name__ == "__main__":
	pickMDA()
