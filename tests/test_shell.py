"""Tests for the sandboxed cross-platform shell tool (chemlagent.shell).

Read-only checks only -- no mkdir here, so the tests never litter the repo.
The sandbox root is bound to the CheMLAgent repo (src/chemlagent/../..), so
the tests assert that behavior relative to this checkout.
"""

from chemlagent import shell as sh


def test_pwd_starts_at_repo_root():
    out = sh.shell('pwd')
    assert out.rstrip().endswith('CheMLAgent'), out


def test_ls_and_ls_l():
    out = sh.shell('ls')
    assert 'src' in out.splitlines() and 'recent_work' in out.splitlines(), out
    out = sh.shell('ls -lt src')
    first = out.splitlines()[0]
    assert first.startswith('d') or first.startswith('-'), out  # long form


def test_find_csvs_and_py():
    out = sh.shell('find src -name "*.py" -maxdepth 2 -type f')
    assert 'shell.py' in out and 'agent.py' in out, out
    # regression: command ENDING in a quoted argument must still parse
    out = sh.shell('find . -maxdepth 1 -type f -name "*.csv"')
    assert ('.csv' in out) or ('no matches' in out), out


def test_head_and_wc():
    out = sh.shell('wc -l src/chemlagent/shell.py')
    lines = int(out.split()[0])
    assert lines > 50, out
    out = sh.shell('head -n 2 src/pyproject.toml')
    assert out.strip(), out


def test_sandbox_escape_refused():
    assert 'project folder denied' in sh.shell('ls /etc')
    assert 'project folder denied' in sh.shell('cd ~')
    assert 'project folder denied' in sh.shell('find .. -name "*.py"')


def test_no_chaining_no_rm():
    assert 'not supported' in sh.shell('ls; rm -rf src')
    assert 'not supported' in sh.shell('rm src')
    assert 'command not supported' in sh.shell('curl https://example.com')


def test_unknown_and_empty():
    assert 'command not supported' in sh.shell('frobnicate src')
    assert 'no command provided' in sh.shell(None)
    assert 'no command provided' in sh.shell('')


def test_print_flag_silent_mode(capsys):
    # The tool banner always prints (repo convention); the flag gates the
    # '[cwd] command' detail line.
    sh.print_flag = False
    sh.shell('pwd')
    assert '[.] pwd' not in capsys.readouterr().out
    sh.print_flag = True
    sh.shell('pwd')
    assert '[.] pwd' in capsys.readouterr().out