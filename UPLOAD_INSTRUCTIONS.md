# Publish this repository on GitHub

1. Visit https://github.com/new while signed in to your GitHub account.
2. Repository name: `field-of-duality-bogoliubov-reproducibility`.
3. Choose **Private** initially if co-author authorization for public release or code licensing remains unconfirmed. Do not select a GitHub-generated README or licence for this first push.
4. Create repository. Then upload this folder's contents (not the enclosing ZIP), or use the git commands below from inside this folder.

```bash
git init
git add .
git commit -m "Add Bogoliubov reproducibility code and saved results"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/field-of-duality-bogoliubov-reproducibility.git
git push -u origin main
```

Replace YOUR_USERNAME with the GitHub account name. A licence may be added later with all rights holders' consent.
