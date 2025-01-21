#!/bin/bash

SESSION="FL-Local"
BASE_DIR="sites"

ADMIN="admin"
SERVER="server"
CLIENTS=( "pct" "msd" "syp" )

tmux_sessions=$(tmux list-session -F "#S")
if [[ " ${tmux_sessions[*]} " =~ " ${SESSION} " ]]
then
  echo "tmux session ${SESSION} already exists, skipping session creation."
else
  # Create new tmux session
  echo "Creating new tmux session ${SESSION}"
  tmux new-session -d -s $SESSION

  # Reserve first window for admin
  window=0
  tmux rename-window -t $SESSION:$window "Admin"
  tmux send-keys -t $SESSION:0 "cd ${BASE_DIR}/${ADMIN}; clear" C-m

  # Create new window for server
  window=1
  tmux new-window -t $SESSION:$window -n "Server"
  tmux send-keys -t $SESSION:$window "cd ${BASE_DIR}/${SERVER}; ./startup/docker.sh" C-m
  tmux send-keys -t $SESSION:$window "./startup/start.sh" C-m

  for client in "${CLIENTS[@]}"; do
    ((window+=1))
    tmux new-window -t $SESSION:$window -n "${client^^}"
    tmux send-keys -t $SESSION:$window "cd ${BASE_DIR}/${client}; ./startup/docker.sh" C-m
    tmux send-keys -t $SESSION:$window "./startup/start.sh" C-m
  done

  # Add an extra window for convinence
  ((window+=1))
  tmux new-window -t $SESSION:$window -n "Work"
fi

# Attach to session
tmux attach -t $SESSION:0

# python3 ./run_fl.py --port=8003 --admin_dir="./${workspace}/${admin_username}" \
#  --username="${admin_username}" --run_number="${run}" --app="${algorithms_dir}/${config}" --min_clients="${n_clients}"

