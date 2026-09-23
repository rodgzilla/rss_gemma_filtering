# home-manager module for rss-filter. Takes the flake's `self` so the default
# package is this flake's own build.
self:
{
  config,
  lib,
  pkgs,
  ...
}:
let
  cfg = config.programs.rss-filter;
  toml = pkgs.formats.toml { };
  configFile = toml.generate "rss-filter.toml" cfg.settings;

  # The command the service runs, also on PATH for manual runs (e.g. `rss-filter-run --dry-run`).
  runner = pkgs.writeShellApplication {
    name = "rss-filter-run";
    runtimeInputs = [ cfg.package ];
    text = ''
      # umap's numba kernels cache next to their module; the store is read-only.
      export NUMBA_CACHE_DIR=${lib.escapeShellArg "${config.xdg.cacheHome}/rss-filter/numba"}
      exec rss-filter \
        --config ${configFile} \
        --state-dir ${lib.escapeShellArg cfg.stateDir} \
        --vault ${lib.escapeShellArg cfg.vault} \
        --feeds ${lib.escapeShellArg cfg.feeds} \
        "$@"
    '';
  };
in
{
  options.programs.rss-filter = {
    enable = lib.mkEnableOption "the embedding-based RSS filter";

    package = lib.mkOption {
      type = lib.types.package;
      default = self.packages.${pkgs.stdenv.hostPlatform.system}.default;
      defaultText = lib.literalMD "this flake's `packages.default`";
      description = "The rss-filter package to run.";
    };

    vault = lib.mkOption {
      type = lib.types.str;
      description = "Obsidian vault root (reads `Daily notes/`, writes `Filtered feed/`).";
    };

    feeds = lib.mkOption {
      type = lib.types.str;
      description = ''
        Subscription list: an RSS Dashboard `data.json` (usually
        `<vault>/.rss-dashboard-data/data.json`) or an OPML export. The
        format is picked from the file extension.
      '';
    };

    stateDir = lib.mkOption {
      type = lib.types.str;
      default = "${config.xdg.stateHome}/rss-filter";
      defaultText = lib.literalExpression ''"''${config.xdg.stateHome}/rss-filter"'';
      description = "Embedding store, seen entries and UMAP cache.";
    };

    settings = lib.mkOption {
      type = toml.type;
      default = { };
      description = "config.toml contents; each key overrides the packaged default.";
    };
  };

  config = lib.mkIf cfg.enable {
    # Packaged defaults, each leaf overridable from the host.
    programs.rss-filter.settings = lib.mapAttrsRecursive (_: lib.mkDefault) (
      builtins.fromTOML (builtins.readFile ../rss_filter/config.toml)
    );

    home.packages = [ runner ];

    # Started on demand (i3 keybinding or `systemctl --user start rss-filter`);
    # no timer, no [Install]. A oneshot has no start timeout by default, so a slow
    # first vault build isn't killed.
    systemd.user.services.rss-filter = {
      Unit.Description = "Score new RSS entries against the Obsidian vault";
      Service = {
        Type = "oneshot";
        ExecStart = lib.getExe runner;
      };
    };
  };
}
