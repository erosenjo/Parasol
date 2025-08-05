{
  description = "Environment for Sketchprobe";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-25.05";
  };

  outputs = { self, nixpkgs }: let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
  in {

    devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
            python3Packages.gym
            python3Packages.torch
        ];
    };

  };
}
