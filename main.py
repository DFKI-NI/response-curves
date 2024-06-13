from config import (
    AbruptConfiguration,
    InsectsConfiguration,
    D3Configuration,
    BNDMvsD3Configuration,
    IncrementalConfiguration,
)
from runner import run


def main():
    run("BNDM vs D3", BNDMvsD3Configuration)
    run("D3", D3Configuration)
    run("Abrupt", AbruptConfiguration)
    run("Incremental", IncrementalConfiguration)
    run("Insects", InsectsConfiguration)


if __name__ == "__main__":
    main()
