{
  lib,
  buildPythonApplication,
  setuptools,
  openai,
  feedparser,
  tqdm,
  numpy,
  umap-learn,
  joblib,
  pytestCheckHook,
  pytest-mock,
}:

buildPythonApplication {
  pname = "rss-gemma-filtering";
  version = "0.1.0";
  pyproject = true;

  # Only what the build and its tests need: not the feed exports or docs.
  # tests/ read mock_vault/ (vault smoke tests) and scripts/ (eval helpers).
  src = lib.fileset.toSource {
    root = ../.;
    fileset = lib.fileset.unions [
      ../pyproject.toml
      ../rss_filter
      ../tests
      ../mock_vault
      ../scripts
    ];
  };

  build-system = [ setuptools ];
  dependencies = [
    openai
    feedparser
    tqdm
    numpy
    umap-learn
    joblib
  ];

  nativeCheckInputs = [
    pytestCheckHook
    pytest-mock
  ];
  # umap imports numba, which wants a writable cache directory.
  preCheck = ''
    export NUMBA_CACHE_DIR=$TMPDIR/numba
  '';

  meta.mainProgram = "rss-filter";
}
