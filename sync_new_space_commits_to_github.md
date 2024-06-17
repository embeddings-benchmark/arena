## How to Sync the Huggingface Space Updates to Github
If someone accidently commits to the HF space instead of to the Github, we can fix it by manually aligning the Github with the space.

0. Clone this Github: `git clone https://github.com/embeddings-benchmark/arena.git`
1. Add the space a git remote: `git remote add space https://huggingface.co/spaces/mteb/arena`
2. Fetch everything on Git LFS: `git lfs fetch --all`
3. Fetch everything normally: `git fetch space`
4. Merge into the Github: `git merge space/main`. If this works, you can commit/push and be done!
5. Verify that the Github codebase online looks like the Space code!

### Dealing with Git LFS errors
Sometimes if a new LFS file was uploaded the space, git complains because it can't find it. If this happens:

0. Locate the files that it is complaining it doesn't have by clicking on the log error it gave you.
1. You'll need to add that file name to the Github's gitattributes locally (or match it with the HF spaces gitattributes file)
2. Upload the file to the repo manually (e.g. copy it into the right spot)
3. Add it to the git history with git add/commit/push it with `git push origin main.
7. Once all the Git LFS stuff is taken care of, you can do `git push --force origin space/main:main` to force push the space repo into the Github repo. Note: this will overwrite history, so be careful!
8. Now they should align! You can check by adding a new file to the Github and making sure it propogates to the space.
