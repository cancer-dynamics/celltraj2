# Publishing Documentation

`celltraj2` uses the same two-step documentation flow as `sitelab` and
`celltraj`.

## Local Build

Install documentation dependencies:

```bash
python -m pip install -e ".[docs]"
```

Build the local HTML docs:

```bash
bash docs/make_docs.sh
```

The generated site is written to:

```text
docs/build/html/
```

## Copy Into CancerDynamics.org

From the `celltraj2` repository root:

```bash
python docs/publish_to_cancerdynamics.py --build
```

Use strict mode before release-style updates:

```bash
python docs/publish_to_cancerdynamics.py --build --strict
```

This replaces the sibling website copy at:

```text
../cancerdynamics-website/public/docs/celltraj2/
```

The publish script excludes Sphinx `_modules`, `_sources`, and `.doctrees`
directories so the deployed docs expose the rendered API and narrative pages,
not raw source files.

After publishing, review, commit, and push the `cancerdynamics-website`
repository through its normal deployment process.
