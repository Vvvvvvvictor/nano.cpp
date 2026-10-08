# 2024 AK8 HLT scale factors

This directory contains the event-level 2024 Zbb trigger payload and the tools
used to derive it from raw NanoAODv15 `Events` trees. Apply the inclusive OR SF
once per MC event, using the highest-pT selected AK8 jet as the query coordinate.
The measurement does not use Zbb output ntuples, trigger objects, an online
ParticleNet requirement, or a separate online bb-tag SF.

## Layout

- `2024/trigger_sf_correctionlib.json`: the soft-drop payload used by
  `configs/run/zbb_2024_v15.yaml`. Running the derivation does not replace it.
- `derive/measure_trigger_sf.py`: measurement, chunk merging, plots and schema inspection.
- `derive/selection.hxx`: ROOT RDataFrame control-region selection.
- `derive/config.json` and `derive/config_gptmass_x2p.json`: soft-drop and X2p configurations.
- `derive/query_correction.py`: checked correctionlib queries.
- `derive/condor.py`: submission, resumption, recovery and full merging.
- `derive/run.sh`: common LCG setup and CLI entry point; `derive/worker.sh` is
  the shared Condor worker entry point.

Historical results, input lists and job records remain under
[`checks/boosted_2024_trigger_sf`](../../checks/boosted_2024_trigger_sf).
They are not executable entry points. In particular, `results_2024_full/`,
`results_2024_gptmass_x2p/` and the `full_run_2024*` directories are preserved.
The independent QCD diagnostics remain in `checks/zbb_2024_trigger_eff/`.

## Inputs and control region

The reference trigger is `HLT_IsoMu24`. The target decision is
`HLT_AK8PFJet500 OR HLT_AK8PFJet380_SoftDropMass30`, evaluated at event level.
An AK8 jet other than the probe may therefore cause the event to pass.

Data comes from all Muon0/Muon1 jobs in
`jobs/zmm_2024_v15_DATA/job_manifest.json` for Run2024C-I. The MC dataset is:

```text
/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM
```

Explicit input files take precedence. For measurement, a matching MC manifest
can be supplied; otherwise MC files are discovered with DAS. Condor submission
accepts newline-separated Data/MC file lists to avoid repeating discovery.
The MiniAODv6 dataset recorded in the configs is diagnostic metadata, not a
measurement input.

The following cuts are fixed in `selection.hxx`:

| Object or requirement | Selection |
| --- | --- |
| Reference muon | Exactly one with pT > 55 GeV, abs(eta) < 2.4, `tightId`, `pfRelIso04_all < 0.15` |
| Additional muon veto | No other muon with pT > 10 GeV, abs(eta) < 2.4, `looseId`, isolation < 0.25 |
| Additional electron veto | No electron with pT > 10 GeV, abs(eta) < 2.5, `cutBased >= 2` |
| AK4 jets | pT > 30 GeV, abs(eta) < 2.4, DeltaR(muon, jet) > 0.4, central jet ID |
| AK4 b tags | At least two selected jets above the configured `Jet_btagUParTAK4B` threshold |
| AK8 probe | Highest-pT jet with pT > 200 GeV, abs(eta) < 2.4, DeltaR(muon, jet) > 0.8, central jet ID |

Central jet ID requires neutral-hadron fraction < 0.99, neutral-EM fraction
< 0.90, charged-hadron fraction > 0.01, charged multiplicity > 0 and total
charged plus neutral multiplicity > 1. No target-trigger cut is applied to the
denominator. Barrel/endcap diagnostics use abs(eta) < 1.5 / >= 1.5.

The AK4 b-tag threshold is configurable and defaults to `0.0246` (2024 UParT
loose). Trigger branch names, binning, mass definition, pileup payload and the
low-statistics threshold are configurable. The muon/electron working points
recorded in the config must match the fixed cuts; changing those entries alone
does not change the selection and is rejected.

## Coordinates, weights and statistical treatment

Both configurations use the original NanoAOD probe `FatJet_pt`. The default
mass is `FatJet_msoftdrop`; the X2p configuration instead uses:

```text
FatJet_mass * (1 - FatJet_rawFactor) * FatJet_globalParT3_massCorrX2p
```

Default bin edges in GeV are:

```text
pT:   200, 250, 300, 350, 380, 400, 450, 500, 550, 600, 700, 850, 1000, 1500
mass: 0, 30, 45, 60, 90, 120, 160, 250, 500
muon pT diagnostics: 55, 60, 70, 90, 120, 200, 400, 1000
```

The accumulation includes the final bin edge. Correctionlib queries use its
own multibinning boundaries and `flow="error"`; coordinates outside the payload
domain raise an error rather than being clamped.

Data has unit weight. The default golden JSON is
`/cvmfs/cms-griddata.cern.ch/cat/metadata/DC/Collisions24/latest/Cert_Collisions2024_378981_386951_Golden.json`.
It is applied before de-duplication by `(run, luminosityBlock, event)`.
Chunk merging repeats de-duplication globally using selected-event caches;
inconsistent duplicate event contents are an error.

MC uses `genWeight * puWeight`. An existing NanoAOD `puWeight` branch is used;
otherwise `Pileup_nTrueInt` is evaluated with `Collisions24_BCDEFGHI_goldenJSON`
from the configured `puWeights_BCDEFGHI.json.gz` under the 2024 LUM metadata
directory on CVMFS. The payload path and SHA-256 are recorded in the report.

Efficiencies retain numerator/denominator covariance through `sumw` and
`sumw2`. Positive Data efficiency SF errors combine Data and MC efficiency
errors in quadrature. A zero Data numerator receives a one-sided 95% upper
interval. Empty bins, invalid efficiencies and zero MC efficiency yield an
invalid SF. Effective entries are `sumw**2 / sumw2`; the default threshold is
10. **Low-statistics bins are flagged, not invalidated:** their status is 2 and
the query helper accepts them. Reports retain individual 500, 380, both and OR
counts and check `N_or = N_500 + N_380 - N_both`.

## Correction interface and analysis use

| Mass configuration | Correction | Inputs |
| --- | --- | --- |
| Soft-drop | `zbb_2024_or_trigger_sf` | `pt`, `msd`, `variation` |
| X2p | `zbb_2024_or_trigger_sf_gptmass_x2p` | `pt`, `gptmass_x2p`, `variation` |

`variation` is `nominal`, `stat_up` or `stat_down`; these are statistical
variations only. Each correction has `<name>_valid` and `<name>_status`
companions taking the two coordinates. Validity is 0/1; status is 0 (invalid),
1 (valid), or 2 (valid with low effective statistics). Invalid SF bins return
`-1`, never a unity weight. The checked query helper refuses invalid bins:

```bash
data/trigger-sf/derive/run.sh query \
  data/trigger-sf/2024/trigger_sf_correctionlib.json 300 60 --variation nominal
```

The Zbb analysis uses the original highest-pT cleaned AK8 pT and soft-drop mass
and applies the SF once per MC event. Its handling of invalid/out-of-range
queries is weight 1 with `fatjetHLTSF_valid=false`; this is analysis behavior,
not a value stored in invalid derivation bins. X2p is an alternative derivation
and does not change the active soft-drop analysis payload.

## Local checks and Condor workflow

Run commands from the repository root. `run.sh` sets up
`LCG_108/x86_64-el9-gcc13-opt`, uses
`X509_USER_PROXY=/afs/cern.ch/user/j/jiehan/x509up_u152815`, and dispatches
measurement subcommands, `query`, or `condor`. Direct Python invocation requires
the same environment. Long measurements and full merges belong on Condor.

```bash
# At most 10,000 raw events from one file per sample.
data/trigger-sf/derive/run.sh measure --max-events 10000 --max-files 1 \
  --no-plots --output-dir /tmp/trigger-sf-check

# Explicit inputs; switch mass definition with --config.
data/trigger-sf/derive/run.sh measure --data-files DATA.root --mc-files MC.root \
  --config data/trigger-sf/derive/config_gptmass_x2p.json \
  --max-events 10000 --max-files 1 --no-plots --output-dir /tmp/trigger-sf-x2p

data/trigger-sf/derive/run.sh inspect --files DATA.root
data/trigger-sf/derive/run.sh plot --report REPORT/trigger_sf.json --output-dir PLOTS
data/trigger-sf/derive/run.sh merge --inputs CHUNK0/trigger_sf.json CHUNK1/trigger_sf.json \
  --output-dir /tmp/trigger-sf-small-merge
```

Condor measurement outputs must be under `/eos/cms/`. Workers compute in
scratch and upload with `xrdcp`; there is no EOS FUSE transport option. The
merge worker also downloads reports and caches with `xrdcp`. Job descriptions
and input lists stay in AFS. File discovery, validation and recovery planning
read existing EOS results on the submission host.

```bash
data/trigger-sf/derive/run.sh condor submit \
  --output-dir jobs/trigger_sf_2024 \
  --results-dir /eos/cms/store/group/phys_jetmet/jiehan/nano_z_2024/trigger_sf_2024 \
  --submit

# Reuse historical input order instead of querying DAS.
data/trigger-sf/derive/run.sh condor submit \
  --data-files-list checks/boosted_2024_trigger_sf/full_run_2024/data_files.txt \
  --mc-files-list checks/boosted_2024_trigger_sf/full_run_2024/mc_files.txt \
  --config data/trigger-sf/derive/config_gptmass_x2p.json \
  --output-dir jobs/trigger_sf_2024_x2p \
  --results-dir /eos/cms/store/group/phys_jetmet/jiehan/nano_z_2024/trigger_sf_2024_x2p
```

Without `--submit`, submission commands only prepare job files. The default
event limit is `-1` (all events); use `--max-events 10000` and one-file input
lists to test the worker first. Tasks contain at most one MC file and balanced
Data chunks; `--chunk-data-files` defaults to 6. Submission directories must be
new. Measurement jobs retry up to three times after their initial attempt.

```bash
data/trigger-sf/derive/run.sh condor resume \
  --submission-dir SUBMISSION --results-dir EOS_RESULTS --submit

# Split missing tasks into smaller chunks after a wall-time failure.
data/trigger-sf/derive/run.sh condor recover \
  --submission-dir SUBMISSION --results-dir EOS_RESULTS \
  --recovery-submission-dir RECOVERY --recovery-results-dir EOS_RECOVERY \
  --data-files-per-task 2 --submit

data/trigger-sf/derive/run.sh condor build-merge \
  --submission-dir SUBMISSION --results-dir EOS_RESULTS \
  --recovery-submission-dir RECOVERY --recovery-results-dir EOS_RECOVERY \
  --merged-submission-dir MERGED_SUBMISSION --merged-results-dir MERGED_RESULTS

data/trigger-sf/derive/run.sh condor merge \
  --submission-dir MERGED_SUBMISSION --results-dir MERGED_RESULTS \
  --output-dir EOS_FINAL --submit
```

`resume` repeats missing tasks at the recorded event limit. `recover` works
with either mass configuration. Present reports with partial input coverage,
wrong configs or missing caches require investigation and are rejected.
`build-merge` requires original plus recovery reports to cover every input
file exactly once. Without recovery, pass the original submission/results
directly to `condor merge`. Use `--local` instead of `--submit` only for a small
merge, with a local output directory.

For historical job records, pass `--config data/trigger-sf/derive/config.json`
(or `data/trigger-sf/derive/config_gptmass_x2p.json`) to `resume`, `recover`, `build-merge` or
`merge` when the recorded config path no longer exists. The file SHA-256 must
match the recorded digest; history is not rewritten and no automatic path
fallback is used. Historical generated worker scripts reference retired
entry points: prepare fresh jobs with the new commands.

## Outputs and validation

Measurement writes `trigger_sf.json`, `trigger_sf.csv`, `trigger_sf.root`,
`trigger_sf_correctionlib.json` and `.json.gz`, `config_used.json`, input lists,
`data_records.npz`/`mc_records.npz`, cutflows and period summaries. Plots include
inclusive/barrel/endcap efficiency, SF and statistical uncertainty maps;
`--no-plots` skips measurement plots. Merging generates plots and diagnostic
outputs. Keep chunk caches alongside their reports: full merging requires them.
Condor outputs include `worker.log`; job lifecycle logs stay beside `.sub` files,
and startup stdout/stderr are stored in the submission's `logs/` directory.

SF reports record selected-event statistics, opened/failed input coverage,
per-run trigger availability, period summaries, muon-pT and AK8 multiplicity
diagnostics. Full-merge metadata records input-file coverage; a finite recorded
event limit still means a test, not full dataset statistics.

NanoAOD does not expose trigger prescales. Verify reference and target prescales
per run period and validate transfer from the semileptonic control region to
Zbb before physics use. The code does not establish ttbar purity or a trigger
plateau. Check cutflows, path availability and the OR overlap identity on a small
sample before full processing. Temporary validation code, logs and outputs
should be removed after testing.
