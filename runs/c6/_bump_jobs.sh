#!/bin/bash
# raise the cell-6 queue to 4 jobs once the cell-0 phase-1 queue has exited
while pgrep -f "c0_queue[.]py --phase 1" >/dev/null; do sleep 60; done
echo 4 > /home/user/gwm_repo/runs/c6/_jobs
echo "bumped to 4 at $(date -u +%FT%TZ)" >> /home/user/gwm_repo/runs/c6/_bump_jobs.log
