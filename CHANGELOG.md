# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Please add your functional changes to the appropriate section in the PR.
Keep it human-readable, your future self will thank you!


## [0.1.12](https://github.com/ecmwf/anemoi-transform/compare/0.1.11...0.1.12) (2025-07-14)


### Features

* Add orography to surface geopotential filter ([#115](https://github.com/ecmwf/anemoi-transform/issues/115)) ([9b7fd8d](https://github.com/ecmwf/anemoi-transform/commit/9b7fd8d74227461e7bb2f4abd1775ea9342da461))
* Add SingleFieldFilter ([#104](https://github.com/ecmwf/anemoi-transform/issues/104)) ([914d1df](https://github.com/ecmwf/anemoi-transform/commit/914d1df883732c12b32cb4f4f0e0389875e044d7))
* Filter migration from anemoi-datasets + testing ([#93](https://github.com/ecmwf/anemoi-transform/issues/93)) ([abfb33d](https://github.com/ecmwf/anemoi-transform/commit/abfb33d20f26604d481099e9ed8a0a4fe4e1f2cd))
* General purpose clipping filter ([#96](https://github.com/ecmwf/anemoi-transform/issues/96)) ([f70b1ab](https://github.com/ecmwf/anemoi-transform/commit/f70b1abc37844046e1f2a7bf9ba60818ccfbecaa))
* Migrate sum filter ([#121](https://github.com/ecmwf/anemoi-transform/issues/121)) ([3b5596e](https://github.com/ecmwf/anemoi-transform/commit/3b5596e37797a645582b6a2b49b5eae4c1391c90))
* Set resolution metadata in filter operation. ([#98](https://github.com/ecmwf/anemoi-transform/issues/98)) ([7844fea](https://github.com/ecmwf/anemoi-transform/commit/7844fea7cf5f849a9dfd7242440eb9721d0ee9f8))


### Bug Fixes

* Docs sphinx dependency python&gt;=3.11 ([#116](https://github.com/ecmwf/anemoi-transform/issues/116)) ([29f6290](https://github.com/ecmwf/anemoi-transform/commit/29f62906b53af2db87520ec2a93406258861e677))
* Fixed typo in error message ([#120](https://github.com/ecmwf/anemoi-transform/issues/120)) ([242c836](https://github.com/ecmwf/anemoi-transform/commit/242c836e6ca4c267755ea143f4d83993ba61471f))

## [0.1.11](https://github.com/ecmwf/anemoi-transform/compare/0.1.10...0.1.11) (2025-05-26)


### Features

* Extend Variable Class ([#94](https://github.com/ecmwf/anemoi-transform/issues/94)) ([dd32ccf](https://github.com/ecmwf/anemoi-transform/commit/dd32ccfcfae54d17b02d82aa869b39dff886df41))
* thermo conversions ([#86](https://github.com/ecmwf/anemoi-transform/issues/86)) ([fce0db5](https://github.com/ecmwf/anemoi-transform/commit/fce0db53c6352fdc39b9dc2f3ff9e2715dd8d279))


### Bug Fixes

* Rodeo delivery ([#101](https://github.com/ecmwf/anemoi-transform/issues/101)) ([54ccd7f](https://github.com/ecmwf/anemoi-transform/commit/54ccd7f95813e8f58353b8593c41ef21218ea1c3))

## [Unreleased](https://github.com/ecmwf/anemoi-utils/transform/0.0.5...HEAD/compare/0.0.9...HEAD)

## [0.0.9](https://github.com/ecmwf/anemoi-utils/transform/0.0.5...HEAD/compare/0.0.8...0.0.9) - 2024-11-01
## [0.1.10](https://github.com/ecmwf/anemoi-transform/compare/0.1.9...0.1.10) (2025-05-06)


### Bug Fixes

* change literal string to variable in `icon_grid` ([#88](https://github.com/ecmwf/anemoi-transform/issues/88)) ([a7459ea](https://github.com/ecmwf/anemoi-transform/commit/a7459ea5862d06d45c4a0f9458bbeecf589a7fa4))

## [0.1.9](https://github.com/ecmwf/anemoi-transform/compare/0.1.8...0.1.9) (2025-04-05)


### Features

* add grid class and factory ([#71](https://github.com/ecmwf/anemoi-transform/issues/71)) ([db0b70a](https://github.com/ecmwf/anemoi-transform/commit/db0b70ad8eb00945d573c69b665fe1859b0889d9))


### Bug Fixes

* grid support ([#82](https://github.com/ecmwf/anemoi-transform/issues/82)) ([c1dc21d](https://github.com/ecmwf/anemoi-transform/commit/c1dc21d487853e5414decf50a637ce04d629ea54))


### Documentation

* fix typo ([#80](https://github.com/ecmwf/anemoi-transform/issues/80)) ([7bb1d2d](https://github.com/ecmwf/anemoi-transform/commit/7bb1d2d027d1b1f84b5945df6c6ad1bdec80ec79))

## [0.1.8](https://github.com/ecmwf/anemoi-transform/compare/0.1.7...0.1.8) (2025-03-31)


### Bug Fixes

* sea ice concentration correctly treated in oras6_clipping ([#78](https://github.com/ecmwf/anemoi-transform/issues/78)) ([703813a](https://github.com/ecmwf/anemoi-transform/commit/703813aebcdd883d6710a4d15674f9f1bde24a56))

## [0.1.7](https://github.com/ecmwf/anemoi-transform/compare/0.1.6...0.1.7) (2025-03-31)


### Documentation

* fix ([#76](https://github.com/ecmwf/anemoi-transform/issues/76)) ([fc8acad](https://github.com/ecmwf/anemoi-transform/commit/fc8acad014a3e1f5179e403171f1ef2c924accd2))

## [0.1.6](https://github.com/ecmwf/anemoi-transform/compare/0.1.5...0.1.6) (2025-03-31)


### Features

* add GRIB flavours ([#72](https://github.com/ecmwf/anemoi-transform/issues/72)) ([a5cb523](https://github.com/ecmwf/anemoi-transform/commit/a5cb523712d8ad5ad48e644524ca43c4bdf73361))

## [0.1.5](https://github.com/ecmwf/anemoi-transform/compare/0.1.4...0.1.5) (2025-03-28)


### Bug Fixes

* missing metadata attribute ([#68](https://github.com/ecmwf/anemoi-transform/issues/68)) ([4f69cb4](https://github.com/ecmwf/anemoi-transform/commit/4f69cb480cc09c1b9b466a81b671b8427c87866d))


### Documentation

* Docathon ([#62](https://github.com/ecmwf/anemoi-transform/issues/62)) ([413665c](https://github.com/ecmwf/anemoi-transform/commit/413665cf8b475bbf673017bca66b9b1360ded4ea))

## [0.1.4](https://github.com/ecmwf/anemoi-transform/compare/0.1.3...0.1.4) (2025-03-24)


### Features

* plugin support ([#65](https://github.com/ecmwf/anemoi-transform/issues/65)) ([7481145](https://github.com/ecmwf/anemoi-transform/commit/7481145c51f4fbf2fdd43a9b8822b18e32b62449))
* Rodeo transform update ([#61](https://github.com/ecmwf/anemoi-transform/issues/61)) ([3e669c0](https://github.com/ecmwf/anemoi-transform/commit/3e669c0207c68b897126e128a76d82a921960522))


### Bug Fixes

* fix regrid arguments ([#67](https://github.com/ecmwf/anemoi-transform/issues/67)) ([1a730cd](https://github.com/ecmwf/anemoi-transform/commit/1a730cd6354cdd00eedc82ec5bf57eea34e8f797))
* return unmodified fields ([#57](https://github.com/ecmwf/anemoi-transform/issues/57)) ([6f0e6f4](https://github.com/ecmwf/anemoi-transform/commit/6f0e6f46506f8eef28219a26e1cfddca6b81793d))
* undetected as dry pixels and meaningful naming ([#59](https://github.com/ecmwf/anemoi-transform/issues/59)) ([b66ce25](https://github.com/ecmwf/anemoi-transform/commit/b66ce25af3f1affc5c8162567a5754dc91b14889))


### Documentation

* links to GitHub ([#66](https://github.com/ecmwf/anemoi-transform/issues/66)) ([10f5c13](https://github.com/ecmwf/anemoi-transform/commit/10f5c138a6c96f1420d476d0171c5f1850cac43b))

## [0.1.3](https://github.com/ecmwf/anemoi-transform/compare/0.1.2...0.1.3) (2025-03-06)


### Features

* Add `icon_refinement_level` filter ([#32](https://github.com/ecmwf/anemoi-transform/issues/32)) ([a2c0711](https://github.com/ecmwf/anemoi-transform/commit/a2c07114c18e6b631401f92519bd760564f8e1ac))
* Clipping strategy to solve issues in the ORAS6 reanalysis ([#56](https://github.com/ecmwf/anemoi-transform/issues/56)) ([179d602](https://github.com/ecmwf/anemoi-transform/commit/179d602f84ad87d5f7d6fe3b7d2348ed74d55a13))
* Matching fields ([#55](https://github.com/ecmwf/anemoi-transform/issues/55)) ([b6f52d6](https://github.com/ecmwf/anemoi-transform/commit/b6f52d6073b3b5de7e44e5e4694b8eeab0b339c2))


### Bug Fixes

* for opera mask ([#40](https://github.com/ecmwf/anemoi-transform/issues/40)) ([a1f8bf8](https://github.com/ecmwf/anemoi-transform/commit/a1f8bf8b49db74d7b79dea0a800d00dabdaa1ba2))


### Documentation

* fix readthedocs ([#51](https://github.com/ecmwf/anemoi-transform/issues/51)) ([42e874b](https://github.com/ecmwf/anemoi-transform/commit/42e874b1020f6d542d6bf9d20ee3c43483c2abcb))
* use new logo ([#42](https://github.com/ecmwf/anemoi-transform/issues/42)) ([aa13ac5](https://github.com/ecmwf/anemoi-transform/commit/aa13ac5b9424d40f5d6bce8279ecbe73292bcc0b))

## 0.1.2 (2025-02-04)

<!-- Release notes generated using configuration in .github/release.yml at main -->

## What's Changed
### Other Changes 🔗
* chore: synced file(s) with ecmwf-actions/reusable-workflows by @DeployDuck in https://github.com/ecmwf/anemoi-transform/pull/30

## New Contributors
* @DeployDuck made their first contribution in https://github.com/ecmwf/anemoi-transform/pull/30

**Full Changelog**: https://github.com/ecmwf/anemoi-transform/compare/0.1.1...0.1.2

## [Unreleased](https://github.com/ecmwf/anemoi-utils/transform/0.0.5...HEAD/compare/0.1.0...HEAD)

### Added

- Add regrid filter
- Added repeat-member #18
- Add `get-grid` command
- Add `cos_sin_mean_wave_direction` filter
- Add `icon_refinement_level` filter

## [0.1.0](https://github.com/ecmwf/anemoi-utils/transform/0.0.5...HEAD/compare/0.0.8...0.1.0) - 2024-11-18

## [0.0.8](https://github.com/ecmwf/anemoi-utils/transform/0.0.5...HEAD/compare/0.0.5...0.0.8) - 2024-10-26

### Added

- Add CONTRIBUTORS.md (#5)

### Changed

- Add more attributes to typed variables
- Fix `__version__` import in init
- Update copyright notice
- Add more methods

## [0.0.1] - Initial Release

### Added

- Project skeleton
