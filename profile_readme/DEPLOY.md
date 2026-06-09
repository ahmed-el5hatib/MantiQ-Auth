# How to Deploy Your GitHub Profile README

To display this README on your public GitHub profile, follow these simple steps:

### Step 1: Create a Special Repository on GitHub
1. Go to your GitHub account and click **New Repository**.
2. Set the **Repository name** to exactly: `ahmed-el5hatib` (your GitHub username).
3. GitHub will display a message: *"You found a secret! ahmed-el5hatib/ahmed-el5hatib is a special repository..."*
4. Make sure the repository is **Public**.
5. Check **Initialize this repository with a README** (or leave it unchecked if you push from command line).
6. Click **Create repository**.

---

### Step 2: Push the README Using Git (Command Line)
If you prefer to push it from the command line, run these commands in your terminal:

```bash
# 1. Create a temporary folder and copy the README
mkdir temp_profile && cd temp_profile
cp ../profile_readme/README.md ./README.md

# 2. Initialize Git
git init
git branch -M main

# 3. Add remote (using your GitHub Profile repository)
git remote add origin https://github.com/ahmed-el5hatib/ahmed-el5hatib.git

# 4. Commit and push
git add README.md
git commit -m "Initial profile readme commit"
git push -u origin main
```

---

### Step 3: Verify Your Profile
Go to your GitHub profile page: `https://github.com/ahmed-el5hatib`
You will see this beautiful, professional README displayed right at the top of your profile page!
