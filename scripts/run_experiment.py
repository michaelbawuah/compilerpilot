from compilerpilot.cli import main
if __name__ == '__main__':
    import sys
    raise SystemExit(main(['experiment', *sys.argv[1:]]))
