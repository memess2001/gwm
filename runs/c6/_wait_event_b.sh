#!/bin/bash
# Keeps this session busy while the queues run; exits on the next queue event or after ~115 min.
c6log=/home/user/gwm_repo/runs/c6/_queue_main.log; c0log=/home/user/gwm_c0/runs/c0/_queue_phase1.log
n6=$(grep -cE "\] (done|FAILED)" $c6log); f0=$(grep -c FAILED $c0log); a0=$(grep -c "all finished" $c0log)
end=$(( $(date +%s) + 115*60 ))
while [ $(date +%s) -lt $end ]; do
  [ $(grep -cE "\] (done|FAILED)" $c6log) -gt $n6 ] && { echo "event: cell 6 run finished"; break; }
  [ $(grep -c FAILED $c0log) -gt $f0 ] && { echo "event: cell 0 job FAILED"; break; }
  [ $(grep -c "all finished" $c0log) -gt $a0 ] && { echo "event: cell 0 phase 1 all finished"; break; }
  pgrep -f "c6_queue[.]py" >/dev/null || { echo "event: cell 6 queue process gone"; break; }
  sleep 30
done
date -u +%FT%TZ; tail -2 $c6log; tail -2 $c0log; echo "c0 done since last restart: $(sed -n '/restart after/h;/restart after/!H;${x;p}' $c0log | grep -cE 'done|FAILED')"
/home/user/gwm_repo/runs/_backup_runs.sh 2>&1 | grep -v -E "^remote:|negotiation|acknowledgments" | tail -1
