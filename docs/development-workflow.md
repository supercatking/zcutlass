# Development Workflow

The WSL checkout is the primary repository. The Windows 11 directory is a mirror
for editor access and handoff convenience.

## Primary Repo

```bash
cd /home/zyz/zcutlass
git status --short
```

Make source, build, test, and benchmark decisions from this WSL path. Before
editing, check status and preserve unrelated changes made by others.

## Windows 11 Mirror

Mirror path:

```text
C:\Users\Admin\Documents\Codex\zcutlass
```

After WSL edits, sync the source mirror from PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File "\\wsl.localhost\Ubuntu\home\zyz\zcutlass\tools\sync_to_host.ps1"
```

The sync is directional from WSL to Windows and excludes `.git`, `build`, and
compiled artifacts.

## GitHub Push Note

Push from the WSL repository after reviewing the exact diff:

```bash
cd /home/zyz/zcutlass
git diff -- README.md docs
git status --short
git push
```

If `git push` blocks on authentication, keep committing locally and push once
WSL GitHub credentials are available.
