# Data and licenses

## What ships with the package

| Data | Contents | API |
|---|---|---|
| ACTIVSg2000 regions | 150 bus pairs, 35 batches of two or three buses, 8 location multisets, exact | `activsg2000(buses)`, `bundled("activsg2000")` |
| ACTIVSg2000 reference results | max-min fair points, price of anarchy and its bound for 25 batches of three and five buses; max-min fair points of all 216 placements of the location game, all computed with `DCOPFRegion` | `reference("activsg2000", "poa_implicit")`, `reference("activsg2000", "location_w")`, `activsg2000_location()` |

Every region file records its provenance:
- the buses and periods;
- the method and tolerance;
- the SHA-256 hash of the input scenario;
- whether counterflow appears in the exact facets.

## Built on your machine: IEEE 14-bus

The IEEE 14-bus regions are not shipped. On first use, `ieee14(buses)` does
four things:
1. It downloads the public case file `case14.m` from MATPOWER's repository
   (release 8.1, pinned). That file is a conversion of the University of
   Washington archive's IEEE Common Data Format file.
2. It checks the file's SHA-256 hash.
3. It builds the scenario in a local cache (`ICGYM_DATA_DIR`, default
   `~/.cache/icgym`), adding the package's own data.
4. It exports and caches the region of any set of buses, in a few seconds
   each.

The package's own data, in `icgym/data/overlays/ieee14.json` (MIT), is:
- the thermal limits, which the case lacks;
- linear generator costs;
- the hourly load multipliers of the five representative days.

The case file supplies the topology, reactances, base loads, and generator
sizes. The rebuilt scenario is fixed byte for byte, and a test checks its file
hashes.

Without network access, download `case14.m` yourself and pass
`case_file=...` to `icgym.datasets.fetch_ieee14`, or set `ICGYM_IEEE14_CASE`.
`icgym.datasets.fetch_ieee14()` returns the scenario folder, for
`icgym-export --data` and `icgym-compare --data`.

The scripts that rebuild the catalogs are `scripts/build_catalog.py` and
`scripts/analyze_catalog.py`, run with the bus specifications in `scripts/`.

## Where the data comes from

- **ACTIVSg2000:**
  - **Source:** the synthetic 2,000-bus Texas grid of the Texas A&M Electric
    Grid Test Case Repository
    (<https://electricgrids.engr.tamu.edu/electric-grid-test-cases/activsg2000/>;
    mirror <https://github.com/caseformat/ACTIVSg2000>).
  - **Scenario:** the regions are computed from a scenario derived from that
    case, with five representative days of hourly demand (120 periods). The
    bus demands are the case's values times a seasonal, daily, and weekly
    multiplier. Generators and branch limits are the case's.
  - **Not real data:** the scenario is synthetic. It is not historical ERCOT
    data, an official model, or a forecast.
- **IEEE 14-bus:**
  - **Source:** the IEEE 14-bus test case of the University of Washington
    Power Systems Test Case Archive.
  - **Scenario:** a representative-day scenario with the same five days and a
    scaled hourly load.

## Licenses and attribution

- **Code:** MIT (see `LICENSE`).
- **ACTIVSg2000-derived data:**
  - **License:** the repository's mirror of the case is licensed under
    Creative Commons Attribution 3.0 Unported (CC BY 3.0). The Texas A&M
    repository describes the case as free for commercial and non-commercial
    use.
  - **Adapted:** the bundled regions are adaptations. The demand profile was
    changed as described above, and the regions are computed results, not the
    case files.
  - **Citation requested:** A. B. Birchfield, T. Xu, K. M. Gegner,
    K. S. Shetye, and T. J. Overbye, "Grid Structural Characteristics as
    Validation Criteria for Synthetic Networks," *IEEE Transactions on Power
    Systems*, vol. 32, no. 4, pp. 3258-3265, July 2017,
    doi:10.1109/TPWRS.2016.2616385.
- **IEEE 14-bus case:** not redistributed. You download it from its source
  on first use. The archive states no license and describes its material as
  provided for research and education, not for commercial gain. Check the
  terms of the sources before commercial use. Our additions are MIT.

## Verification

- **ACTIVSg2000:** every bundled batch matches an independent computation by
  the accompanying paper. The max-min fair points agree to 3e-9 MW and the
  price of anarchy to 4e-12, with no violation of the price-of-anarchy bound.
- **Location game:** the shares give the paper's 3 pure equilibria and 1,617
  improving moves, all raising the sorted shares.
- **IEEE 14-bus:** the regions reproduce the paper's max-min fair points and
  price of anarchy to 0.01 MW.
