# Test Environment Setup and Code Review Request

All of the previous unit tests failures (except the network/requests errors related to go2rtc connection which shouldn't block the logic validation) have been resolved.

- Fixed Pandas version incompatibility.
- Fixed `AttributeError: 'NoneType' object has no attribute 'shape'` by properly checking if frame is `None` before referencing its `shape`.
- Fixed start_time discrepancy bug by converting Pandas DatetimeIndex properly back into UNIX timestamps in `motion_activity()`.
- Fixed the `test_gpu_arg_formatting` test failure by using `"preset-vaapi"` instead of `"hwaccel_vaapi"`.

Ready to create the execution plan.
