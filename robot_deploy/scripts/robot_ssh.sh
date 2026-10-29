#!/usr/bin/env bash
# Run one command on the robot's onboard computer, or open a shell with no arguments.
#   robot_deploy/scripts/robot_ssh.sh 'hostname; uptime'
# This only opens the ssh session. It does not move the robot -- but anything you run through
# it is YOUR responsibility: see robot_deploy/CLAUDE.md "Safety rules" before running controllers.
source "$(dirname "${BASH_SOURCE[0]}")/robot_env.sh" || exit 1
robot_ssh "$@"
