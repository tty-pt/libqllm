## [Unreleased]
### Added
- **OpenCode Integration**: OpenAI-compatible HTTP API server (qllm-serve)
  - `POST /v1/chat/completions` - Streaming and non-streaming chat completions
  - `POST /v1/completions` - Text completions (legacy)
  - `GET /v1/models` - Dynamic model detection
  - `GET /health` - Health check endpoint
- **qllmd Protocol v2.0**:
  - `messages` command for multi-turn conversations with JSON message arrays
  - `info` command for model detection
  - JSON error responses
  - Dynamic buffer allocation for flexible prompt handling
- **System prompt truncation**: Automatically truncates large prompts to fit model context window
- **ThreadingHTTPServer**: Improved concurrency handling
- **Documentation**:
  - `docs/OPENCODE.md` - Complete integration guide (527 lines)
  - `docs/PROTOCOL.md` - Protocol specification (253 lines)
  - README updated with quick start guide

### Changed
- Updated to qmap 0.6.0 for improved pointer stability and allocation reuse
- Updated to ndc 1.0.0 with enhanced network features and better platform separation
- Updated to ndx 0.2.0 with comprehensive test suite and improved documentation
- Simplified qmap usage comments based on v0.6.0 behavior (allocation reuse)
- Improved model cache safety leveraging qmap's pointer stability improvements

### Added
- Optional persistent model cache via QLLM_CACHE_FILE environment variable
- Error handling and validation for cache file paths (length limits, permission checks)
- Integration tests for qmap 0.6.0 pointer stability behavior (5 tests)
- Persistent cache tests with mock file simulation (9 tests)
- Edge case coverage: path validation, empty strings, multiple keys, database isolation
- Enhanced qmap mock with file persistence simulation support
- Comprehensive dependency version documentation in README
- Troubleshooting section in README covering cache and dependency issues
- This CHANGELOG to track changes between releases

### Fixed
- Model cache refcount handling now benefits from qmap 0.6.0 allocation reuse
- Improved pointer safety when updating cache entries
- Test infrastructure: integration tests now properly included in test suite
- qmap mock now correctly handles key updates (was only adding new entries)

### Notes
- qmap pkg-config may show version 0.0.1; actual library is 0.6.0 (reinstall with `cd ../qmap && sudo make install` to fix)
- Backward compatible with previous versions
- Test suite expanded from 100 to 109 tests (107 passing, 2 skipped)

## [0.0.3] - 2026-02-22
### Fixed
- Memory leaks in mocks and qllm
- Improved test coverage with mocks

## Previous Releases
- See git history for earlier changes
