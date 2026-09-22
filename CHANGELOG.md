# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

<!-- insertion marker -->
## [v0.3.1](https://github.com/egagli/easysnowdata/releases/tag/v0.3.1) - 2026-09-22

<small>[Compare with v0.2.0](https://github.com/egagli/easysnowdata/compare/v0.2.0...v0.3.1)</small>

### Fixed

- fix(planetscope): make a real order work end to end; gallery renders a real scene ([bd25c45](https://github.com/egagli/easysnowdata/commit/bd25c45c4212db5b1a5e212dec2fe295a9f1f5d6) by egagli).
- fix(archive): open a scheme-less store path as a local store ([d71b0ee](https://github.com/egagli/easysnowdata/commit/d71b0ee753cef56ec9fdc0dc248b85fda202b5b5) by Eric Gagliano).
- fix(sar): never fuse Sentinel-1 geometries — one raster per relative orbit ([c9fbe53](https://github.com/egagli/easysnowdata/commit/c9fbe53433a9ed8a36edadca1f72cafe67b387d8) by egagli).
- fix(hls): a failed band read raises instead of filling the Dataset with nodata ([101d97f](https://github.com/egagli/easysnowdata/commit/101d97f5aa482ed74c4185486dbc8a3710c4fb26) by egagli).
- fix(auth): log in at URS with the netrc for GDAL reads whenever a password exists ([1b7a1a1](https://github.com/egagli/easysnowdata/commit/1b7a1a13a66f3e71288967e41dcc61723dac5922) by egagli).
- fix(sar): drop the nominal Sentinel-1 heading and the constant 39° incidence angle ([736499e](https://github.com/egagli/easysnowdata/commit/736499e0fcfdef3c1f4bfdc9ea33c5417198806a) by egagli).
- fix(snow): Terra-only MODIS mosaics from Planetary Computer; real NSIDC-0768 file names ([8bbbdbd](https://github.com/egagli/easysnowdata/commit/8bbbdbdbe6d6c11411c867b07f5d578a672233ec) by egagli).
- fix(sar): incidence angle from each track's geometry, with the right sign; OPERA reads work ([d12dc9f](https://github.com/egagli/easysnowdata/commit/d12dc9f37684c9dfb73287c08e0c2472f11254f3) by egagli).
- fix(health): probe Planet through the SDK, send a User-Agent on first-byte probes ([e5aeb82](https://github.com/egagli/easysnowdata/commit/e5aeb82fc28802f3539aaea69c93e3d185b32553) by egagli).
- fix(stations): finish the transfer — format the clients, relabel the probes, refresh the archive ([fb1bf3e](https://github.com/egagli/easysnowdata/commit/fb1bf3e191e27ee7453e23830a99bd6c85731016) by egagli).
- fix(deprecation): the shims promised removal in the release they shipped in ([24e93b5](https://github.com/egagli/easysnowdata/commit/24e93b549e928cb6cfabd103b8ae40f9d1581b07) by egagli).

## [v0.2.0](https://github.com/egagli/easysnowdata/releases/tag/v0.2.0) - 2026-09-17

<small>[Compare with v0.0.26](https://github.com/egagli/easysnowdata/compare/v0.0.26...v0.2.0)</small>

### Fixed

- fix(live): three real bugs behind the failing live tests, one marker for the fourth ([0cbfafb](https://github.com/egagli/easysnowdata/commit/0cbfafb7821980be9ed7101e1dae5f7d839f8387) by egagli).
- fix(providers): spell a local zip as /vsizip/ so Windows can open it ([599a296](https://github.com/egagli/easysnowdata/commit/599a296e0d8f6f95bdfa093f0cc8e642233658f3) by egagli).
- fix(ci): generate the autosummary stubs on a first build, and stop writing COGs in fixtures ([7c3952c](https://github.com/egagli/easysnowdata/commit/7c3952c02f69a88dd978470d54306cd6f2671b95) by egagli).
- fix(docs): drop the minigallery directive, and keep doctrees out of the site ([e051058](https://github.com/egagli/easysnowdata/commit/e051058dc47f1edf474c3a343c5f30d243dc0173) by egagli).
- fix(docs): document a re-exported object once, not per module ([10a9b94](https://github.com/egagli/easysnowdata/commit/10a9b94b75ce151c190725c26f2e2de080c35c1a) by egagli).
- fix(yukon): snowfall is its own type, not precipitation in centimetres ([e89ed0e](https://github.com/egagli/easysnowdata/commit/e89ed0ed3b376b216fd41ab9505f2be0242ae706) by egagli).
- fix(stations): never rescale a depth of snowfall into a depth of water ([37e8aa8](https://github.com/egagli/easysnowdata/commit/37e8aa84c363935677837cf79a83be51ff2a931a) by egagli).
- fix(clients): conform get_metadata for one station, emit hPa from Yukon ([defc053](https://github.com/egagli/easysnowdata/commit/defc05342fb1630206cc226aeedb1967b9071a8b) by egagli).
- fix(awdb): serve TAVG, and prefer it over TOBS for the `temp` type ([87c8ea2](https://github.com/egagli/easysnowdata/commit/87c8ea2d23a9b883cf1997a2ac7903b4d754413f) by egagli).
- fix(stations): three contract gaps the live tier found ([00560f6](https://github.com/egagli/easysnowdata/commit/00560f6106cf4f6d89722174029a4c379ddb1f1e) by egagli).
- fix(snow): stream the NSIDC download with iter_content ([ad1473e](https://github.com/egagli/easysnowdata/commit/ad1473e34d45491390e74075d61bc036cf89a670) by Claude).
- fix(earthaccess): do not treat a half-initialised login as logged in; retry once ([fa5f309](https://github.com/egagli/easysnowdata/commit/fa5f3094dc4c3866a02a14fcbadd4d47e1a6bda1) by egagli).
- fix(modis): fail fast when GDAL has no HDF4 driver (MOD10A1F) ([f506c13](https://github.com/egagli/easysnowdata/commit/f506c1341943ec1d0bb059d8fae89028baf2adc7) by egagli).
- fix(yukon): raise on unknown variables and unsupported intervals ([513bc0c](https://github.com/egagli/easysnowdata/commit/513bc0ca93498e16512e52199b1246726211b1e2) by egagli).
- fix(nve): datetime on hourly records, no silent fallbacks, correct stale docstrings ([c7d499c](https://github.com/egagli/easysnowdata/commit/c7d499c3a4b2d30337e03311b0fe002e7b036551) by egagli).
- fix(awdb): convert all variables to metric, add datetime to hourly records, validate inputs ([bdcda3c](https://github.com/egagli/easysnowdata/commit/bdcda3c257a315b1839366f533af9304a4e2ec4f) by egagli).
- fix(cdec): preserve hourly timestamps, clean SWE priority logic, validate inputs ([8a75dbb](https://github.com/egagli/easysnowdata/commit/8a75dbb50c089a50434103000202de17475ef25a) by egagli).
- fix(databc): wire hourly through get_data, scope negative-value filter, validate interval ([01a1a16](https://github.com/egagli/easysnowdata/commit/01a1a16e10c4f14d3911b43d6b03d5e753d2044f) by egagli).
- fix: correct Nepal station coordinates via detect-and-override, not exclusion ([e6a3e0b](https://github.com/egagli/easysnowdata/commit/e6a3e0bd10c88bfcba6ab3542004d2e165640d56) by egagli).
- fix: correct NVE parameter IDs, response parsing, flags — verified live ([fe202a9](https://github.com/egagli/easysnowdata/commit/fe202a97b6065dd96fd7b43ed8a64035101112e6) by egagli).
- fix: NVE fetch self-diagnoses when /Series matches nothing ([71220bf](https://github.com/egagli/easysnowdata/commit/71220bfcd8a9a0a0ec03f58da9279ac0de84035c) by egagli).
- fix: NVE Observations 404s — request only series that exist, chunked ([ff0f318](https://github.com/egagli/easysnowdata/commit/ff0f318d8ed3742b77947764392f853c65f4b8f3) by egagli).
- fix: NVE Observations uses ReferenceTime not StartDate/EndDate ([c4f8aee](https://github.com/egagli/easysnowdata/commit/c4f8aeee52650259aca0f41162cf6433b652487c) by egagli).
- fix: use correct NVE Observations query param name ([c6b44f5](https://github.com/egagli/easysnowdata/commit/c6b44f5178b8c0de79063f4c69aef30a0a9a386d) by egagli).
- fix: NVE Observations 400 (date format), 429 (rate limit), begin_date ([18d0a51](https://github.com/egagli/easysnowdata/commit/18d0a515769703c651e4f0a2ca8ae29fe465f856) by egagli).
- fix: NVE parameters missing when filtering by ParameterId, CI missing API key ([5deee41](https://github.com/egagli/easysnowdata/commit/5deee415a5793ff175f368ca900b3729af3f338d) by egagli).
- fix: NVE API key auth, Pages permissions, detached HEAD push ([fbbc4d7](https://github.com/egagli/easysnowdata/commit/fbbc4d7a6826ef23c34d8a81357ca1b2c7daeb5b) by egagli).
- fix awdb metadata adaptive batching ([367639a](https://github.com/egagli/easysnowdata/commit/367639a61ecadd583ff2ed28c41c4b082245820f) by egagli).
- fix: restore variables_daily for CDEC/DataBC; update README field docs ([8a256ff](https://github.com/egagli/easysnowdata/commit/8a256ff8431be7bde33d346024762c06d55bec4a) by egagli).

## [v0.0.26](https://github.com/egagli/easysnowdata/releases/tag/v0.0.26) - 2026-09-15

<small>[Compare with v0.0.25](https://github.com/egagli/easysnowdata/compare/v0.0.25...v0.0.26)</small>

### Fixed

- fix(hydro): read Köppen-Geiger file 61012822 via ndownloader.figshare.com ([e78aece](https://github.com/egagli/easysnowdata/commit/e78aeceb347fb56e625f347fe9f848c8c5739521) by egagli).
- fix(earthaccess): explicit login, cloud-hosted MOD10A1F, cache dir, UCLA stats index ([a8c2d3e](https://github.com/egagli/easysnowdata/commit/a8c2d3e73a5929a627eac826b750f80f8305223f) by egagli).
- fix(hydro): fetch GRDC WMO basins with GET into a cache; GET-first health probe ([69ab0e0](https://github.com/egagli/easysnowdata/commit/69ab0e0a49275fdcc7737f53a2ca39f0569af6e6) by egagli).

## [v0.0.25](https://github.com/egagli/easysnowdata/releases/tag/v0.0.25) - 2026-08-25

<small>[Compare with v0.0.24](https://github.com/egagli/easysnowdata/compare/v0.0.24...v0.0.25)</small>

### Added

- add kwargs to functions ([730fa84](https://github.com/egagli/easysnowdata/commit/730fa84e2440c2ca1cd07ef76e528d0f21646d25) by egagli).

### Fixed

- fix EE config stuff ([46d870c](https://github.com/egagli/easysnowdata/commit/46d870c80f622b9a6f34eb94a6b3dceefd919fa1) by egagli).

## [v0.0.24](https://github.com/egagli/easysnowdata/releases/tag/v0.0.24) - 2026-06-15

<small>[Compare with v0.0.23](https://github.com/egagli/easysnowdata/compare/v0.0.23...v0.0.24)</small>

### Changed

- Change git push target to main branch ([cef203d](https://github.com/egagli/easysnowdata/commit/cef203d4e66c990fa8c6f3dd052dabdea2980b4d) by Eric Gagliano).

## [v0.0.23](https://github.com/egagli/easysnowdata/releases/tag/v0.0.23) - 2026-06-15

<small>[Compare with v0.0.22](https://github.com/egagli/easysnowdata/compare/v0.0.22...v0.0.23)</small>

### Fixed

- fix: support EARTHDATA_TOKEN in addition to username/password ([7e879a4](https://github.com/egagli/easysnowdata/commit/7e879a4f792206a906c67df128495d53318e2daf) by Claude).

## [v0.0.22](https://github.com/egagli/easysnowdata/releases/tag/v0.0.22) - 2025-07-28

<small>[Compare with v0.0.21](https://github.com/egagli/easysnowdata/compare/v0.0.21...v0.0.22)</small>

## [v0.0.21](https://github.com/egagli/easysnowdata/releases/tag/v0.0.21) - 2025-07-28

<small>[Compare with v0.0.20](https://github.com/egagli/easysnowdata/compare/v0.0.20...v0.0.21)</small>

### Added

- add snodas code and example ([6012c1c](https://github.com/egagli/easysnowdata/commit/6012c1cf5b6864311cf32763d0d4b352a73bd6f0) by egagli).

## [v0.0.20](https://github.com/egagli/easysnowdata/releases/tag/v0.0.20) - 2025-04-09

<small>[Compare with v0.0.19](https://github.com/egagli/easysnowdata/compare/v0.0.19...v0.0.20)</small>

### Added

- add new wmo basins ([929833b](https://github.com/egagli/easysnowdata/commit/929833b5ef8f6df52a2c989da03844e369f4700b) by egagli).

### Fixed

- fix wmo basin example ([79cd83a](https://github.com/egagli/easysnowdata/commit/79cd83a88b6f5cca9964e48db1a40138fdf13b86) by egagli).

## [v0.0.19](https://github.com/egagli/easysnowdata/releases/tag/v0.0.19) - 2025-03-26

<small>[Compare with v0.0.18](https://github.com/egagli/easysnowdata/compare/v0.0.18...v0.0.19)</small>

### Added

- add new function for local incidence angle calculation via gee ([a4089ef](https://github.com/egagli/easysnowdata/commit/a4089ef5d2861a9c544c5ecca41a34c2e086d56f) by egagli).
- add extra note about ERA5 CDS ([4310aca](https://github.com/egagli/easysnowdata/commit/4310aca4b85dba0682cfc95544e23fe70188b13f) by egagli).

## [v0.0.18](https://github.com/egagli/easysnowdata/releases/tag/v0.0.18) - 2025-02-25

<small>[Compare with v0.0.17](https://github.com/egagli/easysnowdata/compare/v0.0.17...v0.0.18)</small>

## [v0.0.17](https://github.com/egagli/easysnowdata/releases/tag/v0.0.17) - 2025-02-24

<small>[Compare with v0.0.16](https://github.com/egagli/easysnowdata/compare/v0.0.16...v0.0.17)</small>

### Added

- add era5 functionality ([90325e3](https://github.com/egagli/easysnowdata/commit/90325e31323b48a08db2453421472632e8766f0e) by egagli).
- add gee credentials as repo secret so we can test gee functionality ([fccff20](https://github.com/egagli/easysnowdata/commit/fccff20fc5c1fc0a45f7fc6d7ab9d811200a4d83) by egagli).

### Fixed

- fix shapely version ([977e6d6](https://github.com/egagli/easysnowdata/commit/977e6d6c8dac80eac8946e298674b44d92b4f650) by egagli).
- fix end of file space ([c923ffa](https://github.com/egagli/easysnowdata/commit/c923ffa9f58b4fe2dd11fbde58917caa0213f7e7) by egagli).

## [v0.0.16](https://github.com/egagli/easysnowdata/releases/tag/v0.0.16) - 2025-01-30

<small>[Compare with v0.0.15](https://github.com/egagli/easysnowdata/compare/v0.0.15...v0.0.16)</small>

### Added

- add hydroBASINS geometries and GRCD major river basins ([c04673a](https://github.com/egagli/easysnowdata/commit/c04673a4febf77c79b62c76a14aa03c520fe5d33) by egagli).
- add example for nlcd ([7c864cf](https://github.com/egagli/easysnowdata/commit/7c864cf32066049ba641db7fc0a51113eea2e636) by egagli).
- add DOI badge ([10a3640](https://github.com/egagli/easysnowdata/commit/10a36400b12dfad13e18b84468b811edb2a85fc0) by Eric Gagliano).
- Add files via upload ([1a071d6](https://github.com/egagli/easysnowdata/commit/1a071d68e086f2813d17950738d1b779229a60a1) by Eric Gagliano).

## [v0.0.15](https://github.com/egagli/easysnowdata/releases/tag/v0.0.15) - 2025-01-26

<small>[Compare with v0.0.14](https://github.com/egagli/easysnowdata/compare/v0.0.14...v0.0.15)</small>

## [v0.0.14](https://github.com/egagli/easysnowdata/releases/tag/v0.0.14) - 2025-01-14

<small>[Compare with v0.0.13](https://github.com/egagli/easysnowdata/compare/v0.0.13...v0.0.14)</small>

### Fixed

- fix python version in pyproject ([b570db3](https://github.com/egagli/easysnowdata/commit/b570db3d5c33c55a05f810468672f2439b2fa2eb) by egagli).

## [v0.0.13](https://github.com/egagli/easysnowdata/releases/tag/v0.0.13) - 2025-01-11

<small>[Compare with v0.0.12](https://github.com/egagli/easysnowdata/compare/v0.0.12...v0.0.13)</small>

### Removed

- remove cli stuff from pyproject to make conda work ([f134ace](https://github.com/egagli/easysnowdata/commit/f134ace648076d7e2b464b37d47495915ca607dd) by egagli).

## [v0.0.12](https://github.com/egagli/easysnowdata/releases/tag/v0.0.12) - 2025-01-11

<small>[Compare with v0.0.11](https://github.com/egagli/easysnowdata/compare/v0.0.11...v0.0.12)</small>

### Added

- add note to self to add chunks on chili function ([7d25878](https://github.com/egagli/easysnowdata/commit/7d258781b682ec46f5b6d55f3ed3c91b15c3eaa0) by egagli).

### Fixed

- fix beautifulsoup4 and bs4 problem ([3d0079e](https://github.com/egagli/easysnowdata/commit/3d0079e5b359b2090b1126f00dcbd1eefa9f447a) by egagli).

## [v0.0.11](https://github.com/egagli/easysnowdata/releases/tag/v0.0.11) - 2024-11-04

<small>[Compare with v0.0.10](https://github.com/egagli/easysnowdata/compare/v0.0.10...v0.0.11)</small>

## [v0.0.10](https://github.com/egagli/easysnowdata/releases/tag/v0.0.10) - 2024-11-04

<small>[Compare with v0.0.9](https://github.com/egagli/easysnowdata/compare/v0.0.9...v0.0.10)</small>

### Added

- add nlcd data ([94f1fa5](https://github.com/egagli/easysnowdata/commit/94f1fa5c1a76fd4c1c0cd45d51c22b09f47c7862) by egagli).
- add chili, proxy for insolation ([078f47c](https://github.com/egagli/easysnowdata/commit/078f47cd4ae102d4cf9d77e7682d9ba185e5f975) by egagli).
- add high volume api for huc geoms ([e02f1d2](https://github.com/egagli/easysnowdata/commit/e02f1d233390e50ab39ade38b813916b66da8861) by egagli).

### Fixed

- fix nlcd doc string ([c57f5c3](https://github.com/egagli/easysnowdata/commit/c57f5c3afc54fadce2ecf3ca856406a677c8e08e) by egagli).
- fix issue where station named not correctly assigned ([00a39b2](https://github.com/egagli/easysnowdata/commit/00a39b2fa38d54bf1c468f6a0ab285987af86b9f) by egagli).
- fix chili case when no data ([bb436f6](https://github.com/egagli/easysnowdata/commit/bb436f6da029ea8b598a6eebe266692d6f535512) by egagli).
- fix transpose issue with chili data ([a4bf8ce](https://github.com/egagli/easysnowdata/commit/a4bf8ce64221d1f47f1ad58e6cf686c579a4569f) by egagli).

## [v0.0.9](https://github.com/egagli/easysnowdata/releases/tag/v0.0.9) - 2024-09-29

<small>[Compare with v0.0.8](https://github.com/egagli/easysnowdata/compare/v0.0.8...v0.0.9)</small>

### Added

- add s2 comparison notebook ([de52ae2](https://github.com/egagli/easysnowdata/commit/de52ae2f1e3d669f07a4027d8aae291c7489265c) by egagli).
- add koppen geiger data ([6ff576c](https://github.com/egagli/easysnowdata/commit/6ff576cda810333869bed33f76c6949659cedbdf) by egagli).

### Fixed

- fix for hls name change ([7e07270](https://github.com/egagli/easysnowdata/commit/7e07270510c8a089a00154027ee2439c8121d125) by egagli).
- fix docs ([5fac030](https://github.com/egagli/easysnowdata/commit/5fac0302de2e3254dcad43e444175869efa16b8b) by egagli).
- fix issue 7 regarding esa worldcover nodata, plot updates ([b624a19](https://github.com/egagli/easysnowdata/commit/b624a19141913db05410b4645c0830ba1c16f44f) by egagli).
- fix ucla swe reanalysis rotation issue ([950ff2a](https://github.com/egagli/easysnowdata/commit/950ff2a46da9ad99e2863af995d53939a888b2d1) by egagli).
- fix documentation page ([4bbef71](https://github.com/egagli/easysnowdata/commit/4bbef71e2e1789548bad7d6327cedeb682ad53c5) by egagli).
- fix naming ([aa7ee78](https://github.com/egagli/easysnowdata/commit/aa7ee78e6c31addd4dedbc2ef8f6702f9946dc10) by egagli).

### Changed

- change clip to clip_box ([f8e404e](https://github.com/egagli/easysnowdata/commit/f8e404ea07d2252084edd3512f25d835b0068e6e) by egagli).

## [v0.0.8](https://github.com/egagli/easysnowdata/releases/tag/v0.0.8) - 2024-08-19

<small>[Compare with v0.0.7](https://github.com/egagli/easysnowdata/compare/v0.0.7...v0.0.8)</small>

### Fixed

- fix s1, remove geobox option ([627fa53](https://github.com/egagli/easysnowdata/commit/627fa53734a4d85d123bf1f5fefcd4bf48583615) by egagli).

## [v0.0.7](https://github.com/egagli/easysnowdata/releases/tag/v0.0.7) - 2024-08-19

<small>[Compare with v0.0.6](https://github.com/egagli/easysnowdata/compare/v0.0.6...v0.0.7)</small>

## [v0.0.6](https://github.com/egagli/easysnowdata/releases/tag/v0.0.6) - 2024-07-16

<small>[Compare with v0.0.5](https://github.com/egagli/easysnowdata/compare/v0.0.5...v0.0.6)</small>

### Added

- add dependencies needed for earthaccess UCLA snow reanalysis ([300b1cd](https://github.com/egagli/easysnowdata/commit/300b1cd094c99ef3a9649af237993ef67e515770) by egagli).

### Fixed

- fix WY and DOWY functions, fix formatting ([37d1094](https://github.com/egagli/easysnowdata/commit/37d109492bc5f13d517ea3a6467c9da1624b92ea) by egagli).

## [v0.0.5](https://github.com/egagli/easysnowdata/releases/tag/v0.0.5) - 2024-07-03

<small>[Compare with v0.0.4](https://github.com/egagli/easysnowdata/compare/v0.0.4...v0.0.5)</small>

### Added

- add era5 data ([8da612d](https://github.com/egagli/easysnowdata/commit/8da612d32e4158472d4be2d25e93302bad1a26fd) by egagli).
- add new dependencies ([2532664](https://github.com/egagli/easysnowdata/commit/2532664133e7a08623edae43151ad3d7d39a5bb9) by egagli).
- add option to keep s1 data in linear power ([eecd9c7](https://github.com/egagli/easysnowdata/commit/eecd9c7a3306af5eb4d2def1506294f47675fea5) by egagli).
- add huc function ([5e11197](https://github.com/egagli/easysnowdata/commit/5e111979569716da00a576304e973234edad38cf) by egagli).
- add rgb norm to S2 ([8781aac](https://github.com/egagli/easysnowdata/commit/8781aacc2aa55136e5eacce8d38e5ee5ce5d4a9b) by eric).
- add southern hemisphere support to water year and DOWY functions... importantly, decided on what to label WY start/end ([a706757](https://github.com/egagli/easysnowdata/commit/a7067574fcf52b35f17be6dc3a7cc86629bb4451) by eric).

### Changed

- change s1 chunking ([b090f1f](https://github.com/egagli/easysnowdata/commit/b090f1f25ec2544c7e4f3b80dfc3d4e86ecb5dd2) by egagli).

## [v0.0.4](https://github.com/egagli/easysnowdata/releases/tag/v0.0.4) - 2024-04-17

<small>[Compare with v0.0.3](https://github.com/egagli/easysnowdata/compare/v0.0.3...v0.0.4)</small>

## [v0.0.3](https://github.com/egagli/easysnowdata/releases/tag/v0.0.3) - 2024-04-14

<small>[Compare with v0.0.2](https://github.com/egagli/easysnowdata/compare/v0.0.2...v0.0.3)</small>

### Added

- add clip to bbox argument with true as default ([cb11d38](https://github.com/egagli/easysnowdata/commit/cb11d38eb2d2348466a74b0b770ae2efd39f95bf) by eric).
- add mod10a2 attrs ([1d55b6a](https://github.com/egagli/easysnowdata/commit/1d55b6a007df471513edd71dba3f4c1ffba08a95) by eric).

### Fixed

- fix spelling error in pathname ([f78a2a7](https://github.com/egagli/easysnowdata/commit/f78a2a7bc894766188a229eb95be7fcb680872ab) by eric).

## [v0.0.2](https://github.com/egagli/easysnowdata/releases/tag/v0.0.2) - 2024-04-02

<small>[Compare with v0.0.1](https://github.com/egagli/easysnowdata/compare/v0.0.1...v0.0.2)</small>

### Added

- add testing framework ([b6cd9b9](https://github.com/egagli/easysnowdata/commit/b6cd9b9efd3415a65047673b73d275348e78eb07) by eric).
- add ability to skip cache ([33749b0](https://github.com/egagli/easysnowdata/commit/33749b0898fddf4161733694b7964b184ebe3924) by eric).
- add earthaccess ([83e1a23](https://github.com/egagli/easysnowdata/commit/83e1a23939add0762ae84a0dd9f0a005d416c381) by eric).
- add new packages ([305c56d](https://github.com/egagli/easysnowdata/commit/305c56d7aa3a733856a843ddde0e75d7340407d2) by eric).
- add templates for new modules ([3103269](https://github.com/egagli/easysnowdata/commit/3103269d31bc82aaff24ed14b13a9f569d52d762) by --global).
- add all data function ([26cf17c](https://github.com/egagli/easysnowdata/commit/26cf17ce5fd1ab0447801b440aa7c6bed70d4acf) by eric).
- add sentinel 1 functionality ([020dbd9](https://github.com/egagli/easysnowdata/commit/020dbd9e80ec2048c36c049eb40e794bcba23bb7) by eric).
- add sentinel 2 class ([a0b49b1](https://github.com/egagli/easysnowdata/commit/a0b49b1322e94bfef11c2b618eae2ce06c9a1169) by eric).
- add new module ([ab50970](https://github.com/egagli/easysnowdata/commit/ab50970cfc77dec49e1ad06e4964955b01bb1703) by eric).
- add new module for automatic weather stations ([fe32c60](https://github.com/egagli/easysnowdata/commit/fe32c60ece6d7639ef4ab11449d8955c411c87db) by eric).
- add get_esa_worldcover function ([450bf0a](https://github.com/egagli/easysnowdata/commit/450bf0aad1438a0fadadb126b508bb7bcd672602) by eric).
- add env file ([399fa92](https://github.com/egagli/easysnowdata/commit/399fa92b67acc6bffd13650989bf5251ada32731) by eric).

### Fixed

- fix time slice inconsistency ([60e21e9](https://github.com/egagli/easysnowdata/commit/60e21e9874ab88a2405bbc1d3132531b19cd9b95) by eric).
- fix datetime to dowy ([418f45c](https://github.com/egagli/easysnowdata/commit/418f45c3149e8ff910bb54a2be25ee4f9795c083) by eric).
- fix variable name ([8fd6b7f](https://github.com/egagli/easysnowdata/commit/8fd6b7f920d8c9377c5ec2e4d6f7f51114e13b94) by eric).
- fix example ([b18f928](https://github.com/egagli/easysnowdata/commit/b18f9289b197c4bf525ce8054e337ba55b6f06ba) by eric).

### Removed

- remove fluff ([26486e3](https://github.com/egagli/easysnowdata/commit/26486e34781b9d274aff5ca8f8c414ce4585dd69) by eric).

## [v0.0.1](https://github.com/egagli/easysnowdata/releases/tag/v0.0.1) - 2024-03-05

<small>[Compare with first commit](https://github.com/egagli/easysnowdata/compare/eeaf604acd92df02b70d2c410821a5af30b7b27c...v0.0.1)</small>

