#include <Python.h>
#include "nrlmsise00/nrlmsise-00.h"

PyObject *module;

static char module_docstring[] =
	"NRLMSISE-00 wrapper module";
static char gtd7_docstring[] =
	"MSIS Neutral Atmosphere Empircial Model from the surface to lower exosphere.\n\n\
	Parameters\n\
	----------\n\
	input: dict {year=2000, doy=1, sec=0., alt=0., g_lat=0., g_long=0., lst=0.,\n\
			f107A=150., f107=150., ap=4., ap_a=NULL}\n\
		Dictionary containing the NRLMSISE-00 input struct variables,\n\
		defaults to the listed values if missing or invalid.\n\
	flags: list of int (length <= 24), optional\n\
		If shorter that 24 ints, the rest of the switches are set to 0.\n\n\
		Quote from the NRLMSISE-00 source code:\n\
		Switches: to turn on and off particular variations use these switches.\n\
		0 is off, 1 is on, and 2 is main effects off but cross terms on.\n\n\
		Standard values are 0 for switch 0 and 1 for switches 1 to 23. The \n\
		array 'switches' needs to be set accordingly by the calling program. \n\
		The arrays sw and swc are set internally.\n\n\
		switches[i]:\n\
		 i - explanation\n\
		-----------------\n\
		 0 - output in meters and kilograms instead of centimetres and grams\n\
		 1 - F10.7 effect on mean\n\
		 2 - time independent\n\
		 3 - symmetrical annual\n\
		 4 - symmetrical semiannual\n\
		 5 - asymmetrical annual\n\
		 6 - asymmetrical semiannual\n\
		 7 - diurnal\n\
		 8 - semidiurnal\n\
		 9 - daily ap [when this is set to -1 (!) the pointer\n\
		               ap_a in struct nrlmsise_input must\n\
		               point to a struct ap_array]\n\
		10 - all UT/long effects\n\
		11 - longitudinal\n\
		12 - UT and mixed UT/long\n\
		13 - mixed AP/UT/LONG\n\
		14 - terdiurnal\n\
		15 - departures from diffusive equilibrium\n\
		16 - all TINF var\n\
		17 - all TLB var\n\
		18 - all TN1 var\n\
		19 - all S var\n\
		20 - all TN2 var\n\
		21 - all NLB var\n\
		22 - all TN3 var\n\
		23 - turbo scale height var\
		\n\n\
	Returns\n\
	-------\n\
	densities: list\n\
		the NRLMSISE-00 densities:\n\
		d[0] - HE NUMBER DENSITY(CM-3)\n\
		d[1] - O NUMBER DENSITY(CM-3)\n\
		d[2] - N2 NUMBER DENSITY(CM-3)\n\
		d[3] - O2 NUMBER DENSITY(CM-3)\n\
		d[4] - AR NUMBER DENSITY(CM-3)                       \n\
		d[5] - TOTAL MASS DENSITY(GM/CM3) [includes d[8] in td7d]\n\
		d[6] - H NUMBER DENSITY(CM-3)\n\
		d[7] - N NUMBER DENSITY(CM-3)\n\
		d[8] - Anomalous oxygen NUMBER DENSITY(CM-3)\n\n\
		O, H, and N are set to zero below 72.5 km\n\n\
		d[5], TOTAL MASS DENSITY, is NOT the same for subroutines GTD7 \n\
		and GTD7D\n\
		SUBROUTINE GTD7 -- d[5] is the sum of the mass densities of the\n\
		species labeled by indices 0-4 and 6-7 in output variable d.\n\
		This includes He, O, N2, O2, Ar, H, and N but does NOT include\n\
		anomalous oxygen (species index 8).\n\
	temperatures: list\n\
		the NRLMSISE-00 temperatures:\n\
		t[0] - EXOSPHERIC TEMPERATURE\n\
		t[1] - TEMPERATURE AT ALT\n\n\
		t[0], Exospheric temperature, is set to global average for\n\
		altitudes below 120 km. The 120 km gradient is left at global\n\
		average value for altitudes below 72 km.\n\n\
	";
static char gtd7d_docstring[] =
	"MSIS Neutral Atmosphere Empircial Model from the surface to lower exosphere.\n\n\
	This subroutine provides Effective Total Mass Density for output\n\
	d[5] which includes contributions from 'anomalous oxygen' which can\n\
	affect satellite drag above 500 km. See the section 'output' for\n\
	additional details.\n\n\
	Parameters\n\
	----------\n\
	input: dict {year=2000, doy=1, sec=0., alt=0., g_lat=0., g_long=0., lst=0.,\n\
			f107A=150., f107=150., ap=4., ap_a=NULL}\n\
		Dictionary containing the NRLMSISE-00 input struct variables,\n\
		defaults as above.\n\
	flags: list of int (length <= 24), optional\n\
		See Documentation for gtd7().\
		\n\n\
	Returns\n\
	-------\n\
	densities, temperatures: lists\n\
		See documentation for gtd7(), except d[5]:\n\n\
		SUBROUTINE GTD7D -- d[5] is the 'effective total mass density\n\
		for drag' and is the sum of the mass densities of all species\n\
		in this model, INCLUDING anomalous oxygen.\
	";

static PyObject *output_to_TupleList(struct nrlmsise_output output)
{
	PyObject *ret = PyTuple_New(2);
	PyObject *dens = PyList_New(9);
	PyObject *temp = PyList_New(2);
	int i;

	for (i = 0; i < 9; i++)
		PyList_SetItem(dens, i, PyFloat_FromDouble(output.d[i]));
	for (i = 0; i < 2; i++)
		PyList_SetItem(temp, i, PyFloat_FromDouble(output.t[i]));

	PyTuple_SetItem(ret, 0, dens);
	PyTuple_SetItem(ret, 1, temp);

	return ret;
}

static int dict_get_int_default(PyObject *dict, const char *key, int def)
{
	PyObject *val = PyDict_GetItem(dict, PyUnicode_FromString(key));

	if (val && PyLong_Check(val))
		return PyLong_AsLong(val);
	return def;
}
static double dict_get_double_default(PyObject *dict, const char *key, double def)
{
	PyObject *val = PyDict_GetItem(dict, PyUnicode_FromString(key));

	if (val && (PyFloat_Check(val) || PyLong_Check(val)))
		return PyFloat_AsDouble(val);
	return def;
}

static struct nrlmsise_input dict_to_input(PyObject *in_dict)
{
	struct nrlmsise_input ret;
	struct ap_array ap_ar;
	int i;
	int ap_list_size = 0;
	PyObject *ap_list;

	ret.year = dict_get_int_default(in_dict, "year", 2000);
	ret.doy = dict_get_int_default(in_dict, "doy", 1);
	ret.sec = dict_get_double_default(in_dict, "sec", 0.);
	ret.alt = dict_get_double_default(in_dict, "alt", 0.);
	ret.g_lat = dict_get_double_default(in_dict, "g_lat", 0.);
	ret.g_long = dict_get_double_default(in_dict, "g_long", 0.);
	ret.lst = dict_get_double_default(in_dict, "lst", 0.);
	ret.f107A = dict_get_double_default(in_dict, "f107A", 150.);
	ret.f107 = dict_get_double_default(in_dict, "f107", 150.);
	ret.ap = dict_get_double_default(in_dict, "ap", 4.);

	ap_list = PyDict_GetItemString(in_dict, "ap_a");
	if (ap_list) {
		ap_list_size = PyList_Size(ap_list);
		if (ap_list_size > 7) {
			PyErr_WarnEx(PyExc_RuntimeWarning,
				"ap list is too long (> 7), cutting.", 2);
			ap_list_size = 7;
		}
		for (i = 0; i < ap_list_size; i++)
			ap_ar.a[i] = PyFloat_AsDouble(PyList_GetItem(ap_list, i));
		ret.ap_a = &ap_ar;
	} else
		ret.ap_a = NULL;

	return ret;
}
static struct nrlmsise_flags list_to_flags(PyObject *fl_list)
{
	struct nrlmsise_flags ret;
	int i;
	int sw_list_size = PyList_Size(fl_list);

	if (sw_list_size > 24) {
		PyErr_WarnEx(PyExc_RuntimeWarning,
				"nrlmsise flag switches list too long (> 24), cutting.", 2);
		sw_list_size = 24;
	}

	for (i = 0; i < sw_list_size; i++)
		ret.switches[i] = PyLong_AsLong(PyList_GetItem(fl_list, i));

	return ret;
}

static PyObject *nrlmsise00_gtd7(PyObject *self, PyObject *args, PyObject *kwargs)
{
	struct nrlmsise_flags msis_flags = {
		{0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
	struct nrlmsise_output msis_output;
	struct nrlmsise_input msis_input;

	PyObject *input_dict, *flags_list = NULL;
	static char *kwlist[] = {"input", "flags", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|O!", kwlist,
				&PyDict_Type, &input_dict, &PyList_Type, &flags_list)) {
		return NULL;
	}
	msis_input = dict_to_input(input_dict);
	if (flags_list)
		msis_flags = list_to_flags(flags_list);

	gtd7(&msis_input, &msis_flags, &msis_output);

	return output_to_TupleList(msis_output);
}

static PyObject *nrlmsise00_gtd7d(PyObject *self, PyObject *args, PyObject *kwargs)
{
	struct nrlmsise_flags msis_flags = {
		{0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
	struct nrlmsise_output msis_output;
	struct nrlmsise_input msis_input;

	PyObject *input_dict, *flags_list = NULL;
	static char *kwlist[] = {"input", "flags", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|O!", kwlist,
				&PyDict_Type, &input_dict, &PyList_Type, &flags_list)) {
		return NULL;
	}
	msis_input = dict_to_input(input_dict);
	if (flags_list)
		msis_flags = list_to_flags(flags_list);

	gtd7d(&msis_input, &msis_flags, &msis_output);

	return output_to_TupleList(msis_output);
}

static PyMethodDef nrlmsise00_methods[] = {
	{"gtd7", (PyCFunction) nrlmsise00_gtd7, METH_VARARGS | METH_KEYWORDS, gtd7_docstring},
	{"gtd7d", (PyCFunction) nrlmsise00_gtd7d, METH_VARARGS | METH_KEYWORDS, gtd7d_docstring},
	{NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef nrlmsise00_module = {
	PyModuleDef_HEAD_INIT,
	"nrlmsise00",   /* name of module */
	module_docstring, /* module documentation, may be NULL */
	-1,       /* size of per-interpreter state of the module,
				 or -1 if the module keeps state in global variables. */
	nrlmsise00_methods
};


PyMODINIT_FUNC PyInit_nrlmsise00(void)
{
	module = PyModule_Create(&nrlmsise00_module);
	return module;
}

#else

PyMODINIT_FUNC initnrlmsise00(void)
{
	module = Py_InitModule("nrlmsise00", nrlmsise00_methods);
}

#endif
