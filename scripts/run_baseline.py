from compilerpilot.cli import main
if __name__ == '__main__':
    import sys
    raise SystemExit(main(['baseline', *sys.argv[1:]]))
