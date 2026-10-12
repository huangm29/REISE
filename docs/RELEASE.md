# Publication checklist

This directory contains the REISE core-model repository, with FluxXAS as an
optional spectrum-generation companion. The checklist concerns public release;
private development and review can proceed independently.

- [x] Installable Python package and `tr-xas` command.
- [x] Encoder, temporal model and forward-model source/configurations.
- [x] Small frozen encoder/TCN example included in the repository.
- [x] Private local configuration, environments and generated runs excluded.
- [x] Original and distributed source/asset checksums recorded.
- [x] Identified third-party license texts and notices collected.
- [ ] Authors choose a license for their own code and confirm adapted third-party provenance.
- [ ] Authors confirm the included example data and weights may be distributed publicly, and state their terms.

The last two items require author decisions. No MIT/Apache license has been
assigned to the authors' work on their behalf.

## Scope and validation

See [VERIFICATION.md](VERIFICATION.md) for what was actually tested. Complete
training datasets, large FluxXAS weights, all paper figures and complete
scientific retraining reproduction are outside this core release's scope.
They are not prerequisites for running the included example.

The original research directory, coauthor repository and their Git history were
not modified. This candidate has its own directory and excludes old Git history.

## Upload this directory

After the author decisions above, create a GitHub repository using the contents
of this directory, keeping its folder structure. The hidden `.gitignore`,
`.gitattributes` and `.github` files belong in the repository too. Do not upload
the parent TR1X directory, the verification environment or generated run outputs.

Demo files are intentionally allowed by `.gitignore`. Large training data and
private path files remain excluded. The ZIP delivered alongside this directory
contains the same intended repository files, without environment/build output.
