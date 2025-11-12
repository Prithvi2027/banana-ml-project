# GitHub Push Instructions

This file contains recommended repository metadata and step-by-step PowerShell commands to publish the project to GitHub.

Suggested repository details
- Name: `banana-ml-project`
- Description: "Banana Ripeness Classification - End-to-End ML Project: PyTorch model + Flask web app for classifying banana ripeness and estimating days until inedible."
- Topics: `machine-learning`, `pytorch`, `flask`, `image-classification`, `computer-vision`, `banana`
- Visibility: Public (recommended for open datasets), or Private if you want to keep it internal.

Notes before pushing
- This repo includes `data/` and `models/` directories. These can be large. The added `.gitignore` already ignores them, but if `models/best_model.pth` or `data/` are already tracked, you should untrack large files before pushing.

To untrack an already-tracked large file (example `models/best_model.pth`):
```powershell
# Remove from index but keep the file locally
git rm --cached models/best_model.pth
# Commit removal from repo
git commit -m "Remove large model file from tracking; add to .gitignore"
```

Two ways to create a GitHub repo and push:

A) Using GitHub CLI (`gh`) — recommended if you have `gh` installed and authenticated
```powershell
# Initialize repo if needed
git init
git add .
git commit -m "Initial commit: banana-ml-project"

# Create remote repository (interactive) and push
gh repo create YOUR_GITHUB_USERNAME/banana-ml-project --public --source=. --remote=origin --push
```

B) Manual (create repo on github.com, then push)
```powershell
# Initialize and push to remote you set up on github.com
git init
git add .
git commit -m "Initial commit: banana-ml-project"

# Replace <URL> with the repo HTTPS or SSH url shown by GitHub
git remote add origin <URL>
git branch -M main
git push -u origin main
```

If the repo already has commits (you cloned it), just add remote and push:
```powershell
git remote add origin <URL>
git push -u origin main
```

Notes about large files and LFS
- For large model files (>50 MB), use Git LFS:
```powershell
# Install LFS (one-time)
# Windows: follow https://git-lfs.github.com/ or use choco/scoop
git lfs install
# Track pth files
git lfs track "*.pth"
# Add changes and commit
git add .gitattributes
git add models/*.pth
git commit -m "Track model files with Git LFS"
```

Recommended GitHub settings
- Enable Issues and Pull Requests
- Add `README.md` (already present)
- Add `LICENSE` (MIT) which we've added
- Optionally add GitHub Actions CI for linting/tests (create `.github/workflows/` later)

After pushing
- Open a new GitHub issue if you want to track items from the project `TODO` list
- Add a short project description and a screenshot to the repo `README` for visibility

If you want, I can:
- Add a minimal `.github/workflows/python-app.yml` CI workflow to run tests/lint on push
- Help remove large tracked files from history (if you've already committed them)
- Create the GitHub repo using `gh` on your behalf if you provide authentication or run the `gh` command locally

---

If you'd like, I can now:
- (A) Stage and commit the new files here in the workspace, and show you the exact PowerShell commands to run locally to publish (I will not run them without your go-ahead), or
- (B) Create a CI workflow next to run lint/tests automatically after push.

Tell me which next step you prefer.
