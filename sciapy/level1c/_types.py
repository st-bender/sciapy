# common data types for consistency
import numpy as np

_float_type = np.float32
_int_type = np.int32

_limb_data_dtype = [
	("sub_sat_lat", _float_type),
	("sub_sat_lon", _float_type),
	("tp_lat", _float_type),
	("tp_lon", _float_type),
	("tp_alt", _float_type),
	("tp_sza", _float_type),
	("tp_saa", _float_type),
	("tp_los", _float_type),
	("toa_sza", _float_type),
	("toa_saa", _float_type),
	("toa_los", _float_type),
	("sat_sza", _float_type),
	("sat_saa", _float_type),
	("sat_los", _float_type),
	("sat_alt", _float_type),
	("earth_rad", _float_type),
]


def _try_decode(s):
	if hasattr(s, "decode"):
		return s.decode()
	return s
