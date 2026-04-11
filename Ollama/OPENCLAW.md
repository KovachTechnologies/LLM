# Installation

``` bash
ollama launch openclaw
```

This will prompt you to install openclaw.  Doing so resulted in permissions errors.  Fix them by creating a directory for global npm packages.

``` bash
# 1. Create a new global folder in your home directory (you own it)
mkdir -p ~/.npm-global

# 2. Tell npm to use this folder instead of /usr/local
npm config set prefix '~/.npm-global'

# 3. Add the new bin directory to your PATH (so `openclaw` command works)
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
# If you use zsh instead of bash, use this line instead:
# echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.zshrc

# 4. Reload your shell so the change takes effect
source ~/.bashrc
# or: source ~/.zshrc
```

Now install via npm:

``` bash
npm install -g openclaw@latest
```

Still issues.  Need to install nvm:

``` bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
```

``` bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads bash completion
```

``` bash
nvm --version
```

``` bash
nvm install 22
nvm use 22
nvm alias default 22
```

``` bash
node --version   # should now show v22.x.x (not 18.19.1)
npm --version
```

Add to `~/.bashrc`

``` bash
# nvm setup
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"
```

``` bash
npm uninstall -g openclaw   # remove the broken install from v18
npm install -g openclaw@latest
```

# Running Openclaw

``` bash
ollama launch openclaw
```

# Running Gateway

``` bash
openclaw gateway run
```

# Accessing via Browser

http://localhost:18789/#token=<token>


