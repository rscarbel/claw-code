# everything stays under $SCRATCH
export DROOT="$SCRATCH/docker-rootless"
mkdir -p "$DROOT"/{bin,home,config/docker,run,data}
chmod 700 "$DROOT/run"

# read-only checks; these do not modify the system
command -v newuidmap newgidmap
grep "^$(whoami):" /etc/subuid /etc/subgid
cat /proc/sys/kernel/unprivileged_userns_clone 2>/dev/null || true
cat /proc/sys/user/max_user_namespaces 2>/dev/null || true
df -T "$SCRATCH"

# keep Docker files in $SCRATCH
export HOME="$DROOT/home"
export DOCKER_BIN="$DROOT/bin"
export XDG_CONFIG_HOME="$DROOT/config"
export XDG_RUNTIME_DIR="$DROOT/run"
export PATH="$DOCKER_BIN:$PATH"
export DOCKER_HOST="unix://$XDG_RUNTIME_DIR/docker.sock"

# install rootless Docker binaries into $SCRATCH
curl -fsSL https://get.docker.com/rootless | sh

# store Docker images/layers in $SCRATCH too
cat > "$XDG_CONFIG_HOME/docker/daemon.json" <<EOF
{
  "data-root": "$DROOT/data"
}
EOF

# start daemon manually (no systemd)
nohup dockerd-rootless.sh > "$DROOT/dockerd.log" 2>&1 &
sleep 5

docker --version
docker info