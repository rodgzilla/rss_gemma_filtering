{
  description = "Embedding-based RSS filter that writes daily digests into an Obsidian vault";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      forAllSystems =
        f:
        nixpkgs.lib.genAttrs [ "x86_64-linux" "aarch64-linux" ] (
          system: f nixpkgs.legacyPackages.${system}
        );
    in
    {
      packages = forAllSystems (pkgs: {
        default = pkgs.python3Packages.callPackage ./nix/package.nix { };
      });

      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = [
            (pkgs.python3.withPackages (
              ps: with ps; [
                openai
                feedparser
                tqdm
                numpy
                umap-learn
                joblib
                pytest
                pytest-mock
              ]
            ))
            pkgs.llama-cpp # for running llama-server by hand on machines without the NixOS module
          ];
        };
      });
    };
}
