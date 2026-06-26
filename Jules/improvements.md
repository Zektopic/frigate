# Future Improvements

Based on the test runs and codebase review, here are several suggested improvements to implement in the future:

1. **Dependency Management**: Ensure consistent versions of libraries (`pandas`, `numpy`, `peewee`, etc.) are pinned in requirements or `pyproject.toml` to avoid the "ModuleNotFoundError" or "AttributeError" exceptions encountered during the test runs. The environment was missing multiple dependencies initially.
2. **Robust Error Handling for Media API**: Add broader safety checks when accessing `frame.shape` in `frigate/api/media.py`. Returning an empty black frame rather than a 500 error when `frame is None` might be a safer approach for live feeds.
3. **Pydantic Upgrades**: Several tests emitted warnings about `parse_obj_as` being deprecated in Pydantic v2.0 (`PydanticDeprecatedSince20`). Refactoring these to use `TypeAdapter.validate_python()` will future-proof the config loading system.
4. **FastAPI Lifespan Events**: There are deprecation warnings for using `@app.on_event("startup")` in `frigate/api/fastapi_app.py`. Migrating to the newer `lifespan` event handlers will ensure compatibility with future FastAPI versions.
5. **Data Downsampling Edge Cases**: In `frigate/api/review.py`, resampling Pandas `DatetimeIndex` structures back into UNIX timestamps manually required subtracting a specific epoch offset. Using a more native timestamp conversion method, or caching normalized timestamps could improve performance for larger summary queries.
# Future Improvements and Features
## Authentication and Security
- Implement password strength validation for new passwords.
- Consider adding rate limiting on login endpoints to prevent brute-force attacks.

## Optimizations
- Add incremental vacuuming configuration to SQLite instead of full to reduce wear on flash memory.
- Reduce frequency of background processes on low-core APUs.
- Investigate using zero-copy reading in the ffmpeg video ingestion loop.

## Frontend
- Verify input validation across all config forms.
