Laptop-side helpers of lane/trees-apple3 (Apple FAST speed round 3, trees).
They only read steward state; every build and timing runs on the cloud Macs.

  st.sh <id>...            status lines of the ids plus the queue table
  peek.sh <mac> <id>       progress lines of a working or finished job
  fetch.sh <mac> <id>      copy speed.stdout to ~/mojolearn-evidence/trees-apple3/runs/
  waitany.sh <id>...       return when any id is no longer PENDING
  summ.py <stdout>         per cell and arm: median ms, the ms list, digests, ratio

Timing jobs are tools/trees_apple_ab.sh (arms) around tools/trees_apple_speed.sh
(cells), submitted with tools/apple_steward.py submit --kind speed --target <mac>.
